```python
import numpy as np
```


```python
a = np.random.randint(0,100,size=(3,4))
print(a)
```

    [[73 22 78  2]
     [77 56 73 64]
     [83 17 93 23]]
    


```python
# 保存csv文件
np.savetxt("score.csv",a,delimiter=",",header="English,Math",comments="",fmt="%d")
```


```python
# 读取csv
b = np.loadtxt("score.csv",dtype=np.int64,delimiter=",",skiprows = 1)
print(b)
# skiprow跳过多少行
```

    [[73 22 78  2]
     [77 56 73 64]
     [83 17 93 23]]
    

#### numpy独有的保存方式
np.save 和np.load,可以储存多维，另外就是两维


```python
c = np.random.randint(0,100,size=(5,2))
print(c)
```

    [[26 83]
     [85 52]
     [35 10]
     [64 86]
     [97 66]]
    


```python
# numpy.save(frame,array)
np.save("c",c) # 保存
```


```python
# 读取
c1 = np.load("c.npy")
print(c1)
```

    [[26 83]
     [85 52]
     [35 10]
     [64 86]
     [97 66]]
    


```python

```
