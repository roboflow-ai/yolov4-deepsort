## Performing Object Tracking

#### Install Anaconda or miniconda

### Create a virtual environment 

conda create -n myenv <name_of_environment>
e.g. conda create -n myenv mndot

### Activate virtual environment

conda activate <name_of_environment>
e.g. conda activate mndot

### Clone repositories in a single folder
```
git clone https://github.com/roboflow-ai/zero-shot-object-tracking
cd zero-shot-object-tracking
git clone https://github.com/openai/CLIP.git CLIP-repo
cp -r ./CLIP-repo/clip ./clip             // Unix based
robocopy CLIP-repo/clip clip\             // Windows
```
### Install requirements (python 3.7+)

```bash
pip install --upgrade pip
pip install -r requirements.txt
conda install -c conda-forge scipy
```

### Install requirements (anaconda python 3.8)
```
conda install pytorch torchvision torchaudio -c pytorch
conda install ftfy regex tqdm requests pandas seaborn
conda install -c conda-forge scipy
pip install opencv pycocotools tensorflow

```
### Download YOLOv7 weights from

https://github.com/WongKinYiu/yolov7?tab=readme-ov-file

### Run with YOLOv7
```bash
python clip_object_tracker.py --weights models/yolov7.pt --source data/video/fish.mp4 --detection-engine yolov7 --info 

python3 clip_object_tracker_bbox_only_separate_columns.py --weights <path_to_YOLOv7_weights> --conf 0.5 --save-txt --save-conf --name <name_with_which_outputfile_to_be_saved>--source <source_video_path> --detection-engine yolov7 --info

e.g.  python3 clip_object_tracker_bbox_only_separate_columns.py --weights models/yolov7x.pt --conf 0.5 --save-txt --save-conf --name 10.2_PM_NTOR --source /home/marya/Desktop/zero-shot-object-tracking/data/video/10.2_PM_NTOR.mp4 --detection-engine yolov7 --info

```
### THE DETECTIONS ARE ALWAYS SAVED IN THE /runs/detect FOLDER FOLLOWED BY THE PATH NAME THAT YOU HAVE SPECIFIED. YOU CAN ALSO SPECIFY OTHER PARAMETERS BY LOOKING AT THE VARIOUS ARGUMENTS THAT CAN BE PASSED IN THE clip_object_tracker_bbox_only_separate_columns.py FILE OR AS GIVEN BELOW

```
```
### In case you want to loop over videos in a folder, the bash files can be run by updating the path where all the video files are stored using the following command

bash run_all.py 

```
--weights WEIGHTS [WEIGHTS ...]  model.pt path(s)
--source SOURCE                  source (video/image)
--img-size IMG_SIZE              inference size (pixels)
--confidence CONFIDENCE          object confidence threshold                      
--overlap OVERLAP                IOU threshold for NMS
--thickness THICKNESS            Thickness of the bounding box strokes
--device DEVICE                  cuda device, i.e. 0 or 0,1,2,3 or cpu
--view-img                       display results
--save-txt                       save results to *.txt
--save-conf                      save confidences in --save-txt labels
--classes CLASSES [CLASSES ...]  filter by class: --class 0, or --class 0 2 3
--agnostic-nms                   class-agnostic NMS
--augment                        augmented inference
--update                         update all models
--project PROJECT                save results to project/name
--name NAME                      save results to project/name
--exist-ok                       existing project/name ok, do not increment
--nms_max_overlap                Non-maxima suppression threshold: Maximum detection overlap.
--max_cosine_distance            Gating threshold for cosine distance metric (object appearance).
--nn_budget NN_BUDGET            Maximum size of the appearance descriptors allery. If None, no budget is enforced.
--api_key API_KEY                Roboflow API Key.
--url URL                        Roboflow Model URL.
--info                           Print debugging info.
--detection-engine               Which engine you want to use for object detection (yolov7, yolov5, yolov4, roboflow).
```
## Acknowledgements

Huge thanks to:

- [yolov4-deepsort by theAIGuysCode](https://github.com/theAIGuysCode/yolov4-deepsort)
- [yolov5 by ultralytics](https://github.com/ultralytics/yolov5)
- [yolov7 by WongKinYiu](https://github.com/WongKinYiu/yolov7)
- [Deep SORT Repository by nwojke](https://github.com/nwojke/deep_sort)
- [OpenAI for being awesome](https://openai.com/blog/clip/)
