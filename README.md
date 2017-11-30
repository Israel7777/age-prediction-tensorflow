# Age prediction fine tuning

age prediction by fine tuning and transfer learning vgg_face model in keras


### Requiremnets
'''
opencv
'''
'''
keras
'''
'''
tensorflow
'''

* [the_source_blog](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)


### To Train
'''
  To train  = python fine-tune.py --train_dir "data/images/train" --val_dir "data/images/test"
  To test with certain pics =  python predict.py --model "model/2_vggface-fc.h5" --image "test/img.jpg"
  To test with web-cam = python cam.py
'''
