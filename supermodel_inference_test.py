import deeplabcut

video_paths = "<paths>"
superanimal_name = 'superanimal_quadruped'
scale_list = []

deeplabcut.video_inference_superanimal(
    video_paths, superanimal_name, scale_list=scale_list, video_adapt=True, videotype="mp4")
