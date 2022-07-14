# **YOLOv6 How to use ? – Best Tutorial Simple**

*   [English Article](https://inside-machinelearning.com/en/use-yolov6/)
*   [French Article](https://inside-machinelearning.com/utiliser-yolov6/)

YOLOv6 is the 2022 version of YOLO.

**YOLO stands for You Only Look Once. It is a Deep Learning model used for detection on images and videos.**

The first version of YOLO was released in 2016. Since then, frequent updates are made with the latest improvements: faster computation, better accuracy.

## **How to use YOLOv6 ?**

*Please during this tutorial you'll see `&> /dev/null` at the end of any command starting by "!". This is use to hide the command output and make the tutorial more legible. Thus it also hide any error. Feel free to remove it!*

To use YOLOv6, we first need to download the Github repository!

In order to do this, we’ll use the `git clone` command to download it to our Notebook:


```python
!git clone https://dagshub.com/nirbarazida/YOLOv6 &> /dev/null
```

Then, we place ourselves in the folder we just downloaded:


```python
%cd YOLOv6
```

    /content/YOLOv6


Next, we have to install all the necessary libraries to use YOLOv6.

Libraries are the following:

*   torch
*   torchvision
*   numpy
*   opencv-python
*   PyYAML
*   scipy
*   tqdm
*   addict
*   tensorboard
*   pycocotools
*   dvc
*   onnx

Fortunately, only one line of code is needed to install all these dependencies:


```python
!pip install -r requirements.txt &> /dev/null
```

We then want to download the weights of the Neural Network.

With the `git clone` command, we’ve downloaded all the architecture of the Neural Network (layers of the model, functions to train it, use it, evaluate it, …) but to use it, we also need the weights.

**In a Neural Network, the weights are the information obtained by the model during training.**

You can manually download any version of the weights [here](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) and then put the file in the YOLOv6 folder.

Or easily download it with this line of code:


```python
!wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt
```

    --2022-07-01 15:07:58--  https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt
    Resolving github.com (github.com)... 140.82.121.4
    Connecting to github.com (github.com)|140.82.121.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/501076075/190fda7d-6a05-47f4-8c31-404823b9d766?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220701%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220701T150703Z&X-Amz-Expires=300&X-Amz-Signature=5d1838635e7f5190d457a6bd268151e18103f90c4fa9e2c41389a078c5d5b899&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=501076075&response-content-disposition=attachment%3B%20filename%3Dyolov6n.pt&response-content-type=application%2Foctet-stream [following]
    --2022-07-01 15:07:58--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/501076075/190fda7d-6a05-47f4-8c31-404823b9d766?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220701%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220701T150703Z&X-Amz-Expires=300&X-Amz-Signature=5d1838635e7f5190d457a6bd268151e18103f90c4fa9e2c41389a078c5d5b899&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=501076075&response-content-disposition=attachment%3B%20filename%3Dyolov6n.pt&response-content-type=application%2Foctet-stream
    Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 9817788 (9.4M) [application/octet-stream]
    Saving to: ‘yolov6n.pt.1’
    
    yolov6n.pt.1        100%[===================>]   9.36M  36.6MB/s    in 0.3s    
    
    2022-07-01 15:07:58 (36.6 MB/s) - ‘yolov6n.pt.1’ saved [9817788/9817788]
    


*At the time I’m writing those lines, YOLOv6 has just been released. Updates may occur and the weights may change, as well as the URL of their repository. If the link no longer works, check [this Github](https://github.com/meituan/YOLOv6) for the latest version.*

One last thing before using the template: upload your image!

**Either a single image, or several in a folder (YOLOv6 can handle several images at once).**

Feeling lazy ? You can simply upload our test image in one line of code :


```python
!wget https://raw.githubusercontent.com/tkeldenich/Tutorial_YOLOv6/main/man_cafe.jpg
```

    --2022-07-01 15:07:59--  https://raw.githubusercontent.com/tkeldenich/Tutorial_YOLOv6/main/man_cafe.jpg
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.109.133, 185.199.108.133, 185.199.111.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.109.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 47111 (46K) [image/jpeg]
    Saving to: ‘man_cafe.jpg.1’
    
    man_cafe.jpg.1      100%[===================>]  46.01K  --.-KB/s    in 0.003s  
    
    2022-07-01 15:07:59 (16.5 MB/s) - ‘man_cafe.jpg.1’ saved [47111/47111]
    


Let's see what it looks like:


```python
from PIL import Image
im = Image.open('man_cafe.jpg')
im
```




![png](Readme_files/Readme_15_0.png)



FINALLY, we can use YOLOv6!

To do this, we’ll call the infer.py file in the tools folder.

The python code in this file will run the detection for us.

**We only need to specify a few important parameters, in our case :**

*   weights we are using: `--weights yolov6n.pt`
*   image on which we want to apply the detection: `--source ./man_cafe.jpg`


```python
!python tools/infer.py --weights yolov6n.pt --source ./man_cafe.jpg
```

    Namespace(agnostic_nms=False, classes=None, conf_thres=0.25, device='0', half=False, hide_conf=False, hide_labels=False, img_size=640, iou_thres=0.45, max_det=1000, name='exp', project='runs/inference', save_img=True, save_txt=False, source='./man_cafe.jpg', weights='yolov6n.pt', yaml='data/coco.yaml')
    Save directory already existed
    Loading checkpoint from yolov6n.pt
    
    Fusing model...
      0% 0/1 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2228.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    100% 1/1 [00:00<00:00,  3.77it/s]
    Results saved to runs/inference/exp


If you use your own image or folder, just change this last part to `./your_path`.

## **Understand the results**

**You'll find the detection output in the folder** `YOLOv6/runs/inference/exp/`

Here is our result :


```python
im_res1 = Image.open('./runs/inference/exp/man_cafe.jpg')
im_res1
```




![png](Readme_files/Readme_21_0.png)



First of all, we can see that YOLOv6 has a very good detection capacity. It even detects the person right behind our main character.

**You can see that it even indicates the confidence with which it detects objects/people. Most obvious objects are detected with a high confidence threshold. But we can see on the left that YOLOv6 detects a bottle. Here the model fails. But it fails with a very low confidence threshold.**

It detects with 34% confidence that there is a bottle on the left table. This means that YOLOv6 is not at all sure about this prediction.

Maybe we can simply remove this prediction?

**Actually it’s possible when running YOLOv6.**

Remember that we have defined some parameters. The mandatory parameters are the weights and the source (image path), but there are also default parameters that you can change to suit your needs.

**There are other parameters, for example:**

*   Confidence threshold: conf-thres (default value: 0.25)
*   Intersection over Union threshold: iou-thres (default value: 0.45)

The confidence threshold defines the minimum confidence value at which the detection must be kept.

By default, if a detection has a confidence threshold lower than 25%, it will be deleted.

Let’s change the confidence threshold to 0.35 :


```python
!python tools/infer.py --weights yolov6n.pt --source ./man_cafe.jpg --conf-thres 0.35
```

    Namespace(agnostic_nms=False, classes=None, conf_thres=0.35, device='0', half=False, hide_conf=False, hide_labels=False, img_size=640, iou_thres=0.45, max_det=1000, name='exp', project='runs/inference', save_img=True, save_txt=False, source='./man_cafe.jpg', weights='yolov6n.pt', yaml='data/coco.yaml')
    Save directory already existed
    Loading checkpoint from yolov6n.pt
    
    Fusing model...
      0% 0/1 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2228.)
      return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
    100% 1/1 [00:00<00:00,  3.77it/s]
    Results saved to runs/inference/exp


Result :


```python
im_res2 = Image.open('./runs/inference/exp/man_cafe.jpg')
im_res2
```




![png](Readme_files/Readme_25_0.png)



Here we see that the bottle has not been kept.

Use YOLOv6, make assumptions for your project and set the confidence threshold that suits your need 😉

## **Changing weights**

Now you understand a bit more what YOLOv6 is.

Let’s go deeper.

There are different versions of YOLOv6 weights.

We have used a large version of these weights.

The large weights mean three things compared to the other versions:

1.   Better results (or at least a better understanding of the complexity)
2.   Slower computation speed
3.   More memory space used

**But what if you have space and time constraints?**

Well, ou can use other versions of these weights:

*   **Small**: https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt
*   **Medium**: https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.pt
*   **Large**: https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt

Again, you can download them manually [here](https://github.com/meituan/YOLOv6/releases/tag/0.1.0) and then put them into your folder, or use `wget` and specify the version you want:


```python
!wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt
```

    --2022-07-01 15:08:03--  https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt
    Resolving github.com (github.com)... 140.82.121.3
    Connecting to github.com (github.com)|140.82.121.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://objects.githubusercontent.com/github-production-release-asset-2e65be/501076075/7af0d5c3-a8e5-456e-ae8c-29b7fc22753e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220701%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220701T150704Z&X-Amz-Expires=300&X-Amz-Signature=3720829a2ada17ccbc901f6d30c19d063e932ddbf15e7ade832489945d6f3386&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=501076075&response-content-disposition=attachment%3B%20filename%3Dyolov6s.pt&response-content-type=application%2Foctet-stream [following]
    --2022-07-01 15:08:03--  https://objects.githubusercontent.com/github-production-release-asset-2e65be/501076075/7af0d5c3-a8e5-456e-ae8c-29b7fc22753e?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20220701%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20220701T150704Z&X-Amz-Expires=300&X-Amz-Signature=3720829a2ada17ccbc901f6d30c19d063e932ddbf15e7ade832489945d6f3386&X-Amz-SignedHeaders=host&actor_id=0&key_id=0&repo_id=501076075&response-content-disposition=attachment%3B%20filename%3Dyolov6s.pt&response-content-type=application%2Foctet-stream
    Resolving objects.githubusercontent.com (objects.githubusercontent.com)... 185.199.109.133, 185.199.108.133, 185.199.111.133, ...
    Connecting to objects.githubusercontent.com (objects.githubusercontent.com)|185.199.109.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 38102844 (36M) [application/octet-stream]
    Saving to: ‘yolov6s.pt’
    
    yolov6s.pt          100%[===================>]  36.34M  12.2MB/s    in 3.0s    
    
    2022-07-01 15:08:07 (12.2 MB/s) - ‘yolov6s.pt’ saved [38102844/38102844]
    


Depending on the weight version you’ve chosen, you will have :

*   **Small**: yolov6s.pt
*   **Medium**: yolov6t.pt
*   **Large**: yolov6n.pt

To use it, replace `weight_file` here:


```python
!python tools/infer.py --weights yolov6s --source ./man_cafe.jpg
```

    Namespace(agnostic_nms=False, classes=None, conf_thres=0.25, device='0', half=False, hide_conf=False, hide_labels=False, img_size=640, iou_thres=0.45, max_det=1000, name='exp', project='runs/inference', save_img=True, save_txt=False, source='./man_cafe.jpg', weights='yolov6s', yaml='data/coco.yaml')
    Save directory already existed
    Traceback (most recent call last):
      File "tools/infer.py", line 108, in <module>
        main(args)
      File "tools/infer.py", line 103, in main
        run(**vars(args))
      File "/usr/local/lib/python3.7/dist-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
        return func(*args, **kwargs)
      File "tools/infer.py", line 95, in run
        inferer = Inferer(source, weights, device, yaml, img_size, half)
      File "/content/YOLOv6/yolov6/core/inferer.py", line 33, in __init__
        self.model = DetectBackend(weights, device=self.device)
      File "/content/YOLOv6/yolov6/layers/common.py", line 259, in __init__
        assert isinstance(weights, str) and Path(weights).suffix == '.pt', f'{Path(weights).suffix} format is not supported.'
    AssertionError:  format is not supported.


Result:


```python
im_res3 = Image.open('./runs/inference/exp/man_cafe.jpg')
im_res3
```




![png](Readme_files/Readme_33_0.png)



**Now you know how to use YOLOv6!**

If you want to know more about [Computer Vision](https://inside-machinelearning.com/en/category/computer-vision-en/), feel free to check [our category dedicated to this topic.](https://inside-machinelearning.com/en/category/computer-vision-en/)

See you soon in a next post 😉
