import wandb
import yaml
from jsonargparse import Namespace as JsonNamespace
from jsonargparse import namespace_to_dict
from loguru import logger

from data_juicer.config import init_configs, merge_config
from data_juicer.core import Analyser
from data_juicer.core import Executor as DjExecutor
from data_juicer.core.sandbox.factories import (data_evaluator_factory,
                                                mode_infer_executor_factory,
                                                model_evaluator_factory,
                                                model_train_executor_factory)
from tools.hpo.execute_hpo_3sigma import modify_recipe_k_sigma


class SandBoxWatcher:
    """
    Basic Watcher class to manage interested results, and manage the experiment
    within the sanbox based on WandB UI and it's utilities.
    """

    def __init__(self, dj_cfg):
        """
        Initialize the watcher with a reference to an executor instance.
        """

        # the web-ui and experiment versioning is based on WandB
        project_name = dj_cfg.project_name
        sweep_config = dj_cfg.sweep_config
        self.dj_cfg = dj_cfg

        self.wandb_run = wandb.init(project=project_name)
        if (sweep_config is not None and 'metric' in sweep_config
                and 'name' in sweep_config['metric']):
            self.object_name_in_hpo = sweep_config['metric']['name']
        else:
            self.object_name_in_hpo = None
        self.logged_res = {}

    def query(self, res_name: str):
        """
        Query the result from the logged_res.
        """
        return self.logged_res.get(res_name)

    def watch(self, res, res_name: str = ''):
        """
        Flatten the result in dot structure and log it into WandB.
        """
        if isinstance(res, dict):
            for key, value in res.items():
                # getting the left nodes of the given res dictionary.
                if isinstance(value, dict):
                    self.watch(value, f'{res_name}.{key}')
                else:
                    self.logged_res[f'{res_name}.{key}'] = value
                    if self.object_name_in_hpo == f'{res_name}.{key}':
                        # Ensuring float results for HPO experiments
                        value = float(value)
                    self.wandb_run.log({f'{res_name}.{key}': value})
        else:
            self.logged_res[res_name] = res
            if res_name == self.object_name_in_hpo:
                res = float(res)
            self.wandb_run.log({res_name: res})

    def setup_sweep(self, sweep_config: dict = None, project_name: str = None):
        """
        Setup and start a new WandB sweep.
        """
        if sweep_config is None:
            sweep_config = self.dj_cfg.sweep_config
        if project_name is None:
            project_name = self.dj_cfg.project_name
        sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
        if (sweep_config is not None and 'metric' in sweep_config
                and 'name' in sweep_config['metric']):
            self.object_name_in_hpo = sweep_config['metric']['name']
        return sweep_id

    def watch_cfgs(self, cfgs: list[(dict, str)] = None):
        """
        Watch the configuration of the experiment.
        """
        merged_cfgs = {}
        if cfgs is not None:
            for cfg, cfg_prefix in cfgs:
                if isinstance(cfg, JsonNamespace):
                    converged_cfg = namespace_to_dict(cfg)
                elif isinstance(cfg, dict):
                    converged_cfg = cfg
                else:
                    raise ValueError(
                        f'Expected dict or JsonNamespace, got {type(cfg)}')
                for key, val in converged_cfg.items():
                    merged_cfgs[f'{cfg_prefix}.{key}'] = val
        else:
            merged_cfgs = namespace_to_dict(self.dj_cfg)

        wandb.config.update(merged_cfgs)


class SandBoxExecutor:
    """
    This SandBoxExecutor class is used to provide a sandbox environment for
        exploring data-model co-designs in a one-stop manner with fast feedback
         and tiny model size, small data size, and high efficiency.

        It plays as a middleware maintains the data-juicer's data executor,
        a model processor (training and inference), and an auto-evaluator,
        where the latter two ones are usually from third-party libraries.

    """

    def __init__(
        self,
        dj_cfg=None,
        model_infer_cfg=None,
        model_train_cfg=None,
        data_eval_cfg=None,
        model_eval_cfg=None,
    ):
        """
        Initialization method.

        :param dj_cfg: configuration of data-juicer,
            for data recipe and sandbox (e.g., HPO and leveraged tools).
        :param model_infer_cfg: configuration of
            an integrated model inference utility.
        :param model_train_cfg: configuration of
            an integrated model training utility.
        :param data_eval_cfg: configuration of an
            integrated auto-evaluation utility for data.
        :param model_eval_cfg: configuration of an
            integrated auto-evaluation utility for model.

        """
        self.dj_cfg = init_configs() if dj_cfg is None else dj_cfg

        self.watcher = SandBoxWatcher(self.dj_cfg)
        self.watcher.watch_cfgs([
            (dj_cfg, 'data_juicer'),
            (model_infer_cfg, 'model_infer'),
            (model_train_cfg, 'model_train'),
            (data_eval_cfg, 'data_eval'),
            (model_eval_cfg, 'model_eval'),
        ])

        self.data_executor = DjExecutor(self.dj_cfg)
        self.model_infer_executor = mode_infer_executor_factory(
            model_infer_cfg)
        self.model_trainer = model_train_executor_factory(model_train_cfg)
        self.data_evaluator = data_evaluator_factory(data_eval_cfg)
        self.model_evaluator = model_evaluator_factory(model_eval_cfg)

        # default jobs to probe, refine_recipe, execution and evaluation for
        # interested data and model within the sandbox
        self.probe_jobs = []
        self.refine_recipe_jobs = []
        self.execution_jobs = []
        self.evaluation_jobs = []

        self.register_default_jobs()

    def hook_probe_via_analyzer(self, res_name: str):
        # probe the data via Analyser
        logger.info('Begin to analyze data')
        analyser = Analyser(self.dj_cfg)
        analyser.run()
        analyser_res = analyser.overall_result
        self.watcher.watch(analyser_res, res_name)

    def hook_probe_via_model_infer(self, res_name: str):
        # probe the model (calling inference sub-pipeline) based on
        # original data, such that we know what is the "hard" data for
        # the model and how to process the data accordingly
        if (self.model_infer_executor is not None
                and self.data_evaluator is not None):
            sampled_data = self.data_executor.sample_data(
                sample_ratio=self.dj_cfg.data_probe_ratio,
                sample_algo=self.dj_cfg.data_probe_algo,
            )
            res_type, infer_res = self.model_infer_executor.run(sampled_data)
            measure_on_infer_res = self.data_evaluator.run(res_type, infer_res)
            self.watcher.watch({res_name: measure_on_infer_res})

    def hook_refine_recipe_via_k_sigma(self):
        # use k-sigma strategy to modify the data recipe
        if self.dj_cfg.path_k_sigma_recipe is not None:
            modify_recipe_k_sigma(self.dj_cfg,
                                  self.watcher.query('analyser_res'),
                                  self.dj_cfg.path_k_sigma_recipe)

    def hook_refine_recipe_via_model_feedback(self):
        # use model-feedback-based strategy to modify the data recipe,
        # e.g., more mapper on the "hard" or "sensitive" data, those were
        # ranked by user-interested measurement after model inference
        if self.dj_cfg.path_model_feedback_recipe is not None:
            # modify_recipe_model_feedback(
            #     self.dj_cfg,
            #     self.watcher.query("measure_on_infer_res"),
            #     self.dj_cfg.path_model_feedback_recipe)
            raise NotImplementedError('Not implemented yet.')

    def hook_process_data(self):
        # basic routine to process data, users can customize this freely
        logger.info('Begin to process the data with given dj recipe')
        self.data_executor.run()

    def hook_train_model(self, **kwargs):
        # basic routine to train model via the processed data,
        # users can customize this freely
        logger.info('Begin to train the model with given model config')
        self.model_trainer.run(**kwargs)

    def hook_evaluate_data(self, **kwargs):
        # basic routine to evaluate the given data,
        # users can customize this freely
        logger.info('Begin to evaluate the data with given evaluator config')
        self.data_evaluator.run(**kwargs)

    def hook_evaluate_model(self, **kwargs):
        # basic routine to evaluate the given model,
        # users can customize this freely
        logger.info('Begin to evaluate the model with given evaluator config')
        self.model_evaluator.run(kwargs)

    def register_default_jobs(self):
        self.probe_jobs.append((self.hook_probe_via_analyzer, {
            'res_name': 'analysis_ori_data'
        }))
        self.probe_jobs.append((self.hook_probe_via_model_infer, {
            'res_name': 'analysis_ori_model'
        }))

        self.refine_recipe_jobs.append(
            (self.hook_refine_recipe_via_k_sigma, None))
        self.refine_recipe_jobs.append(
            (self.hook_refine_recipe_via_model_feedback, None))

        self.execution_jobs.append((self.hook_process_data, None))
        self.execution_jobs.append((self.hook_train_model, None))

        self.evaluation_jobs.append((self.hook_probe_via_analyzer, {
            'res_name': 'analysis_processed_data'
        }))
        self.evaluation_jobs.append((self.hook_probe_via_model_infer, {
            'res_name': 'analysis_trained_model'
        }))
        self.evaluation_jobs.append((self.hook_evaluate_data, {
            'res_name': 'eval_data'
        }))
        self.evaluation_jobs.append((self.hook_evaluate_model, {
            'res_name': 'eval_model'
        }))

    def run(self):
        """
         Running the sandbox pipeline at once or in HPO style.
        """
        if self.dj_cfg.hpo_config is not None:
            # execute_hpo_wandb contains running one_trail with HPO scheduler
            self.execute_hpo_wandb()
        else:
            self.one_trial()

    def one_trial(self):
        """
        Running the sandbox pipeline at once.
         Users can flexibly conduct some steps of the whole sandbox pipeline
          according to their own need and configuration. The watcher will
          automatically track the results in terms of data, model and specified
          evaluation metrics to the watcher.

        """
        if self.watcher.object_name_in_hpo is not None:
            # merge the new hyper-parameters produced by HPO scheduler
            self.dj_cfg = merge_config(self.dj_cfg, wandb.config)
            self.watcher.watch_cfgs(self.dj_cfg)

        # ====== Data & model probe ======
        for probe_hook, args in self.probe_jobs:
            probe_hook(args)

        # ====== Data-model recipes iteration based on probe results ======
        for refine_job, args in self.refine_recipe_jobs:
            refine_job(args)

        # ====== Data processing & model training ======
        for exec_hook, args in self.execution_jobs:
            exec_hook(args)

        # ====== Evaluation on processed data or trained model ======
        for eval_hook, args in self.evaluation_jobs:
            eval_hook(args)

    def execute_hpo_wandb(self):
        """
        Running the sandbox pipeline in HPO style.
         Users can flexibly conduct some steps of the whole sandbox pipeline
          according to their own need and configuration. The watcher will
          automatically track the results in terms of data, model and specified
          evaluation metrics to the watcher.
        """
        with open(self.dj_cfg.hpo_config) as file:
            sweep_configuration = yaml.safe_load(file)
            sweep_id = self.watcher.setup_sweep(sweep_configuration)
            wandb.agent(sweep_id,
                        function=self.one_trial,
                        count=sweep_configuration['sweep_max_count']
                        if 'sweep_max_count' in sweep_configuration else None)