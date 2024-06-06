Usage Example:

python liveness_test.py --video_path test_shubham.mp4 --output_folder  outs --skip 10
python liveness_test_softmax.py --video_path test_nilesh.mp4 --output_folder  outs --skip 10

Reads test_shubham.mp4 by skipping every 10 frames and saving the processed frames in 'outs' folder



# set video_path to 0 for using webcam
python liveness_test.py --video_path 0 --output_folder  outs

# set output_path to 0 for displaying frames
python liveness_test.py --video_path test.mp4 --output_folder  0
