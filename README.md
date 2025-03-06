Thank you for participating in this experiment. Below are the instruction to
run the experiment.


### Requirements

The following python packages are needed, any version should work.

```
opencv-python
numpy
matplotlib
scikit-image
```

### Run

Pick a username in all lower case e.g. `skocour` and please run all 5 questions

```
python3 runme.py --username <username> --question Q1
python3 runme.py --username <username> --question Q2
python3 runme.py --username <username> --question Q3
python3 runme.py --username <username> --question Q4
python3 runme.py --username <username> --question Q5
```


### Study Questions & Answer Guide:

1️⃣ **"What object was there?"** (7 options)
   - Press **a**, **b**, **c**, **d**, **e**, **f**, or **g** corresponding to the object you believe was removed from the bounding box area

2️⃣ **"Was an object removed from this image?"** (y/n)
    (does it look like any object from the scene was fully removed?)
   - Press **y** = Yes, an object was removed
   - Press **n** = No, no object was removed

3️⃣ **"Was an object removed from this area?"** (y/n)
    (does it look like any object from the the bounding box area was fully removed?)
   - Press **y** = Yes, an object was removed within the bounding box area
   - Press **n** = No, no object was removed within the bounding box area

4️⃣ **"In which image is the object best removed?"** (4 options)
    (best means you can not recognize what the removed object was, and the rest of the scene is not affected)
   - Press **a**, **b**, **c**, or **d** corresponding to the image where the object is best removed

5️⃣ **"In which image is the object best removed?"** (2 options)
    (best means you can not recognize what the removed object was, and the rest of the scene is not affected)
   - Press **l** = LEFT image has the object better removed
   - Press **r** = RIGHT image has the object better removed