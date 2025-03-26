## 配置环境说明
```python
python 3.9.5
pytorch 1.11.0
cuda 11.3
在4卡4090下面的环境（rgt）
```

```
cd Model_code
pip install -r requirements.txt


环境安装的时候需要注意的依赖包
pip install setproctitle
pip install timm
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install numpy==1.23.5

python setup.py develop --no_cuda_ext
```
