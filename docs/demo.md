# Demo code
## Inference TRG on video

Please run the following code in the root directory (TRG-Release/).

If you have a video that you want to run inference on, place it in the example directory and execute the code. This will produce a rendered video.
 
```
python trg_demo_video.py --video_path example/demo.mp4 \
  --frame_rate 30 \
  --fiter_threshold 0.9 \
  --do_render \
  --model_path checkpoint/trg_240717/checkpoint-30/state_dict.bin
```