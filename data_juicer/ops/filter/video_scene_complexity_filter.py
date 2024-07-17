import numpy as np
from jsonargparse.typing import PositiveInt, ClosedUnitInterval

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields,StatsKeys

from ..base_op import OPERATORS,Filter

from data_juicer.utils.model_utils import get_model, prepare_model
from ..mapper.video_tagging_from_frames_mapper import \
    VideoTaggingFromFramesMapper

from ..op_fusion import LOADED_VIDEOS,INTER_SAMPLED_FRAMES

OP_NAME = 'video_scene_complexity_filter'

with AvailabilityChecking(['scipy','SentenceTransformer'], OP_NAME):
    from scipy.cluster.hierarchy import fcluster, linkage

@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
@INTER_SAMPLED_FRAMES.register_module(OP_NAME)
class VideoSceneComplexityFilter(Filter):
    # Filter videos with insufficient tags
    def __init__(self,
                 hf_text_similarity_model='sentence-transformers/all-MiniLM-L6-v2',
                 contain: str = 'any',
                 frame_sampling_method: str = 'all_keyframes',
                 frame_num: PositiveInt = 3,
                 threshold: ClosedUnitInterval = 0.5,
                 min_tags: PositiveInt = 3,
                 min_tag_categories: PositiveInt = 2,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method [{frame_sampling_method}] is not '
                f'supported. Can only be one of ["all_keyframes", "uniform"].')
        if contain not in ['any', 'all']:
            raise ValueError(f'the containing type [{contain}] is not '
                             f'supported. Can only be one of ["any", "all"].')
        self.contain = contain

        self._accelerator = 'cuda'

        self.tagging_producer = VideoTaggingFromFramesMapper(
            frame_sampling_method=frame_sampling_method,
            frame_num=frame_num,
        )
        self.model_key = prepare_model(
            model_type = 'SentenceTransformer',
            pretrained_model_name_or_path = hf_text_similarity_model)
        
        self.threshold = threshold
        self.min_tags = min_tags
        self.min_tag_categories = min_tag_categories

    def compute_stats(self, sample, rank=None, context=False):
        # wheather get tags already
        if Fields.video_frame_tags not in sample:
            sample = self.tagging_producer.process(sample, rank, context)

        video_tags = sample[Fields.video_frame_tags][0]
        
        tag_count = len(video_tags)        
        sample[Fields.stats][StatsKeys.video_tag_numbers] = tag_count

        if tag_count <= 1:
            sample[Fields.stats][StatsKeys.video_tag_categories] = tag_count
            return sample

        model = get_model(self.model_key,rank=rank)

        # 编码标签
        embeddings = model.encode(video_tags)
        
        # 计算相似度矩阵
        similarities = model.similarity(embeddings, embeddings)
        
        # 将相似度转换为距离
        distances = 1 - similarities
        
        # 使用层次聚类
        linkage_matrix = linkage(distances, method='complete')
        
        # 根据阈值切分聚类
        clusters = fcluster(linkage_matrix, t=1-self.threshold, criterion='distance')
        
        # 计算唯一类别的数量
        num_categories = len(np.unique(clusters))

        sample[Fields.stats][StatsKeys.video_tag_categories] = num_categories

        print(f"sample after{sample}:")
        return sample

    def process(self, sample, rank=None):
        video_tag_numbers = sample[Fields.stats][StatsKeys.video_tag_numbers]
        video_tag_categories = sample[Fields.stats][StatsKeys.video_tag_categories]

        condition_tag_numbers = video_tag_numbers >= self.min_tags
        condition_tag_categories = video_tag_categories >= self.min_tag_categories
        if self.contain == 'any':
            return condition_tag_numbers or condition_tag_categories
        else:
            return condition_tag_numbers and condition_tag_categories