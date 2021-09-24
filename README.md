# Continual Learning on the Edge with TensorFlow Lite
This repository contains the source-code and APK developed for our paper: [Continual Learning on the Edge with TensorFlow Lite](https://arxiv.org/abs/2105.01946)
* <strong>Offline Experiments</strong> includes the source-code necessary to reproduce the results presented in section
<i>"Enhancing TensorFlow Lite Capabilities with Continual Learning"</i>.
* <strong>Android Demo App</strong> includes the source-code of our Continual Learning application built on top off TensorFlow Lite.
* <strong>cldemo.apk</strong> can be used to install and test our demo-app on an Android smartphone device.
  

## On-device Demos
In order to compare their performance in real-world scenarios,
both Transfer Learning and Continual Learning models were deployed on a Samsung
Galaxy S10, Android device using TensorFlow Lite.

* A video demonstrating the training and inference during the on-device experiments 
for all scenarios can be seen [here](https://www.youtube.com/watch?v=OUvWhQouSu8&ab_channel=LearningAgentsandRobotsMRG).
* A video demonstrating the training and testing of the CL model under non-ideal conditions
can be seen [here](https://www.youtube.com/watch?v=mVI1ob55vZw).
  
You can [download](https://gitlab.com/riselear/public/continual-learning-on-the-edge-with-tensorflow-lite/-/blob/master/cldemo.apk) the Android demo-app APK
and install it on your own device. Compatible with Android 8.0 and later versions.

## How to run Offline Experiments
* To reproduce the results presented in section <i>"Enhancing TensorFlow Lite Capabilities with Continual Learning"</i>,
it is first necessary to download the [CORe50 dataset](https://vlomonaco.github.io/core50/).
* You also need to change the <strong>root in experiments.py</strong> to the path where your dataset is located.
* All experiments bellow measure accuracy over time based on the CORe50 NICv2 - 391 benchmark

#### Options
* Compare the Transfer Learning model with the Continual Learning model<br>
  ``
python controller.py --exp_tl_vs_cl
``
* Compare FIFO with Random Replacement of old samples in the replay buffer<br>
  ``
python controller.py --exp_sample_replacement
``
* Compare different replay buffer sizes<br>
  ``
python controller.py --exp_buffer_size
``
