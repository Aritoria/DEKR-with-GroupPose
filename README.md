# Image_analysis
![Image text](https://github.com/Aritoria/Image_analysis/blob/main/GroupPose/util/image.png)

To Run The Code

1 Configure the environment according to the requirements in the configuration file.  
To learn more detailed information, please refer to the requirement.txt file.


2 Enter the virtual environment and import the COCO dataset.  

 
3 Execute the following command for training.  
```python -m torch.distributed.run --nproc_per_node=(Your GPU Number!!!) --master_port 29579 main.py -c config/grouppose.py --coco_path <path/to/coco/> --output_dir <path/to/output> ```

4 Execute the following command for testing.  
```python -m torch.distributed.run --nproc_per_node=(Your GPU Number!!!) --master_port 29579 main.py -c config/grouppose.py --backbone resnet50 --coco_path <path/to/coco> --output_dir <path/to/output> --resume <checkpoint> --eval```  

 (Note: the dataset path and output result path should be specified by the user.)  




