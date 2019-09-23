# TIMERS
This is a sample implementation of "[TIMERS: Error-Bounded SVD Restart on Dynamic Networks](http://cuip.thumedialab.com/papers/TIMERS.pdf)"(AAAI 2018).

### Requirements
```
MATLAB (MATLAB 2017a works fine for me)
```

### Usage
##### Example Usage
```
See TIMERS_Sample.m for a sample run on the syntactic network
```
##### Functions
```
TIMERS_Sample.m: a sample run on the syntactic networks, see annotations for detail
RefineBound.m: calculate the lower bound of the objective function, our main results (see Eq. (13) in our paper)
Random_Com.m: generate a random graph with communities forming (see RANDOM-Com Dataset in our paper)
Obj.m: input a similarity matrix S and two embeddings U,V, return || S - U * V ||_F^2, using a trick to reduce memory cost
Obj_SimChange.m: returns the new objective function when only S changes

```
### Cite
If you find this code useful, please cite our paper:
```
@inproceedings{zhang2018timers,
  title={TIMERS: Error-Bounded SVD Restart on Dynamic Networks},
  author={Zhang, Ziwei and Cui, Peng and Pei, Jian and Wang, Xiao and Zhu, Wenwu},
  booktitle={Proceedings of the 32nd AAAI Conference on Artificial Intelligence},
  year={2018},
  organization={AAAI}
}
```