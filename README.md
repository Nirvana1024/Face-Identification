# Face-Identification
use this model to identify someone

use new_real.py  to start the project

in the working directory 
open the terminal 
input
```bash
new_real.py 
```
##  1.collect data 
you are then required to input the name of the person who is to be identifided
(at present,this model can be used to identify two persons,but you can choose to input only one person’s name)

```bash
plz input two names(q to quit): LiChao
['LiChao']
data//LiChao// 创建成功
```

the camera of your computer will start to capture your face
and pictures will be counted(200 training face data in total)


![](https://github.com/Nirvana1024/Face-Identification/raw/master/captured_faces.png)


after the collection of the training human face pictures
a file named ‘data’ will be created which is used to store the captured training face data
![](https://github.com/Nirvana1024/Face-Identification/raw/master/data_file.png)

then input ‘q’ to quit
```bash
plz input two names(q to quit):q

```


## 2. train the model
once input 'q' to quit,the training process will start imediately
and the training will begin (10 epoch in total)
![](https://github.com/Nirvana1024/Face-Identification/raw/master/train.png)


## 3. test your model 
finally, after the model have been trained, the camera can identify the target person
![](https://github.com/Nirvana1024/Face-Identification/raw/master/predict.png)
