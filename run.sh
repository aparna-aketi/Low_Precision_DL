# deep-squeeze experiments no compression on CIFAR-10 with ResNet-20 over a 8 node directed ring
python trainer_ds.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --k=0.0 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --k=0.0 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

# deep-squeeze experiments 99% compression on CIFAR-10 with ResNet-20 over a 8 node directed ring
python trainer_ds.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.099 --compressor='sparsify' --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp


# deep-squeeze experiments no compression on CIFAR-10 with VGG-11 over a 8 node directed ring
python trainer_ds.py --arch=vgg11 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=vgg11 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=vgg11 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_ds.py --arch=vgg11 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp


# CHOCO-SGD experiments no compression on CIFAR-10 with ResNet-20 over a 8 node directed ring
python trainer_choco.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=1  --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

# CHOCO-SGD experiments 99% compression on CIFAR-10 with ResNet-20 over a 8 node directed ring
python trainer_choco.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --depth=20 --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.099 --compressor='sparsify' --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --depth=20 --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=cifar10 --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp


# CHOCO-SGD experiments no compression on Imagenette with ResNet-18 over a 8 node directed ring
python trainer_choco.py --arch=resnet --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=1 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=1 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --skew=0.6 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=1 --k=0.0 --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --skew=0.6 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=1 --k=0.0 --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

# CHOCO-SGD experiments 90% compression on Imagenette with ResNet-18 over a 8 node directed ring
python trainer_choco.py --arch=resnet --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --skew=0.0 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=0.01 --k=0.099 --compressor='sparsify' --quantized_train=0 --port=3005 
python dict_to_csv.py --save-dir=save_temp

python trainer_choco.py --arch=resnet --skew=0.8 --world_size=8 --batch-size=256 --lr=0.1 --dataset=imagenette --qgm=0 --eta=0.01 --k=0.99 --compressor='sparsify' --quantized_train=1 --port=3005 
python dict_to_csv.py --save-dir=save_temp






