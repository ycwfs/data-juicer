from jsonargparse.typing import PositiveInt

from data_juicer.utils.availability_utils import AvailabilityChecking
from data_juicer.utils.constant import Fields
from data_juicer.utils.mm_utils import (SpecialTokens,extract_key_frames,
                                        extract_video_frames_uniformly,
                                        load_data_with_context, load_video)
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Mapper
from ..op_fusion import LOADED_VIDEOS

OP_NAME = 'video_caption_from_tag_mapper'

with AvailabilityChecking(
    ['torch', 'git+https://github.com/xinyu1205/recognize-anything.git','random'],
        OP_NAME):
    import ram  # noqa: F401
    import torch
    import random

    # avoid hanging when calling recognizeAnything in multiprocessing
    torch.set_num_threads(1)


@OPERATORS.register_module(OP_NAME)
@LOADED_VIDEOS.register_module(OP_NAME)
class VideoCaptionFromTagMapper(Mapper):
    """Mapper to generate video caption from tags extract by video.
    """

    def __init__(self,
                 frame_sampling_method: str = 'all_keyframes',
                 frame_num: PositiveInt = 3,
                 keep_original_sample: bool = True,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param frame_sampling_method: sampling method of extracting frame
            images from the videos. Should be one of
            ["all_keyframes", "uniform"].
            The former one extracts all key frames (the number of which depends
            on the duration of the video) and the latter one extract specified
            number of frames uniformly from the video.
            Default: "all_keyframes".
        :param frame_num: the number of frames to be extracted uniformly from
            the video. Only works when frame_sampling_method is "uniform". If
            it's 1, only the middle frame will be extracted. If it's 2, only
            the first and the last frames will be extracted. If it's larger
            than 2, in addition to the first and the last frames, other frames
            will be extracted uniformly within the video duration.
        :param args: extra args
        :param kwargs: extra args
        """
        super().__init__(*args, **kwargs)
        if frame_sampling_method not in ['all_keyframes', 'uniform']:
            raise ValueError(
                f'Frame sampling method [{frame_sampling_method}] is not '
                f'supported. Can only be one of ["all_keyframes", "uniform"].')
        self.model_key = prepare_model(
            model_type='recognizeAnything',
            pretrained_model_name_or_path='tag2text_swin_14m.pth',
            input_size=384)
        self.frame_sampling_method = frame_sampling_method
        self.frame_num = frame_num
        from ram import get_transform
        self.transform = get_transform(image_size=384)

    def process(self, sample, rank=None, context=False):
        # check if it's generated already
        if Fields.video_frame_tags in sample:
            return sample

        # there is no video in this sample
        if self.video_key not in sample or not sample[self.video_key]:
            sample[Fields.video_frame_tags] = []
            return sample

        # load videos
        loaded_video_keys = sample[self.video_key]
        sample, videos = load_data_with_context(sample, context,
                                                loaded_video_keys, load_video)

        model = get_model(self.model_key, rank=rank)

        for _, value in enumerate(loaded_video_keys):
            video = videos[value]

            # extract frame images
            if self.frame_sampling_method == 'all_keyframes':
                frames = extract_key_frames(video)
            elif self.frame_sampling_method == 'uniform':
                frames = extract_video_frames_uniformly(video, self.frame_num)
            else:
                frames = []

            frame_tensor = torch.stack([
                self.transform(frame.to_image()) for frame in frames
            ]).to(next(model.parameters()).device)
            with torch.no_grad():
                captions, tags = model.generate(frame_tensor,
                                              tag_input=None,
                                              max_length=100,
                                              return_tag_predict=True)
        max_length = 0
        longest_tag_indices = []
        
        for i, tag_string in enumerate(tags):
            tag_count = len(tag_string.split('|'))
            if tag_count > max_length:
                max_length = tag_count
                longest_tag_indices = [i]
            elif tag_count == max_length:
                longest_tag_indices.append(i)
        
        chosen_index = random.choice(longest_tag_indices)
        longest_tag = [i.strip() for i in tags[chosen_index].split("|")]
        corresponding_caption = captions[chosen_index]

#{'videos': ['/data01/resized/18FTZ.mp4'], 'text': '<__dj__video> <|__dj__eoc|>', '__dj__stats__': {'video_tag_numbers': 77, 'video_tag_categories': 58}, '__dj__video_frame_tags__': [['man', 'shirt', 'doorway', 'room', 'stool', 'carpet', 'floor', 'squat', 'mat', 'cloth', 'paper', 'red', 'stand', 'woman', 'scroll', 'material', 'pen', 'writing', 'home appliance', 'chair', 'person', 'paper towel', 'toilet paper', 'dark', 'boy', 'bend', 'tool', 'wear', 'kitchen', 'book', 'polo shirt', 'bible', 'sit', 'strip', 'black', 'check', 'food', 'read', 'write', 'corridor', 'wash', 'bathroom', 'magnet', 'mark', 'pencil', 'plaid', 'elevator', 'exhaust hood', 'sink', 'stare', 'open', 'job', 'pet', 'pillow', 'remote', 'door', 'hallway', 'bulletin board', 'can', 'pink', 'alley', 'design', 'napkin', 'notebook', 'roll', 'dog', 'floor mat', 'cat', 'girl', 'wall', 'clipboard', 'pillar', 'extinguisher', 'laptop', 'lamp', 'brush', 'mouth']]}
        sample[Fields.video_frame_tags] = longest_tag
        sample[self.text_key] = f"{SpecialTokens.video} {corresponding_caption} {SpecialTokens.eoc}"
        print(sample)

        return sample
