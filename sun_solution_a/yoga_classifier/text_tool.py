import re

class surya_name(object):
    def __init__(self):
        self.split = "-|\."
    def __call__(self, text):
        x = re.split(self.split, text)
        # print(x)
        return x[1]+"-"+x[2]

name = surya_name()
result = name("a-MountainPose-Samasthiti.csv")
print(result)
