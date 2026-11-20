# mammal, box
python train.py -debug 1 -dataset mammal -train_non_basic_percent 0 -dim 2 -margin 0.001 -epoch 400 -lr 0.01 -eval_method energy -loss_type contrastive -batch_size 32 -mode box -add_denominator non -use_threshold 1 -hybrid 0 -lamda 1.0 -p 1

python train.py -debug 1 -dataset mammal -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 400 -lr 0.005 -eval_method energy -loss_type contrastive -batch_size 32 -mode box -add_denominator non -use_threshold 1 -hybrid 0 -lamda 0.5 -p 1

# mammal, ball
python train.py -debug 1 -dataset mammal -train_non_basic_percent 0 -dim 2 -margin 0.001 -epoch 400 -lr 0.01 -eval_method energy -loss_type contrastive -batch_size 32 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 1.0 -p 2

python train.py -debug 1 -dataset mammal -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 400 -lr 0.01 -eval_method energy -loss_type contrastive -batch_size 32 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 0.5 -p 2

----------------------------------------------------------------------------------------------------------------------------
## noun, box
python train.py -debug 1 -dataset noun -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 400 -lr 0.005 -eval_method energy -loss_type contrastive -batch_size 2048 -mode box -add_denominator non -use_threshold 1 -hybrid 0 -lamda 1.0 -p 1

python train.py -debug 1 -dataset noun -train_non_basic_percent 0 -dim 10 -margin 0.001 -epoch 400 -lr 0.005 -eval_method energy -loss_type contrastive -batch_size 2048 -mode box -add_denominator non -use_threshold 1 -hybrid 0 -lamda 1.0 -p 1


## noun, ball
python train.py -debug 1 -dataset noun -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 400 -lr 0.01 -eval_method energy -loss_type contrastive -batch_size 2048 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 1.0 -p 2

python train.py -debug 1 -dataset noun -train_non_basic_percent 0 -dim 10 -margin 0.001 -epoch 400 -lr 0.01 -eval_method energy -loss_type contrastive -batch_size 2048 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 1.0 -p 2


----------------------------------------------------------------------------------------------------------------------------
## MCG, box
python train.py -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 500 -lr 0.0005 -eval_method energy -loss_type contrastive -batch_size 1024 -mode box -add_denominator in -use_threshold 1 -hybrid 0 -lamda 1.0 -p 2

python train.py -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 10 -margin 0.001 -epoch 500 -lr 0.0005 -eval_method energy -loss_type contrastive -batch_size 2048 -mode box -add_denominator non -use_threshold 1 -hybrid 1 -lamda 0.0 


## MCG, ball
python train.py -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 500 -lr 0.01 -eval_method energy -loss_type contrastive -batch_size 1024 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 0.0 

python train.py -debug 1 -dataset MCG -train_non_basic_percent 0 -dim 10 -margin 0.001 -epoch 500 -lr 0.005 -eval_method energy -loss_type contrastive -batch_size 1024 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 0.0 

----------------------------------------------------------------------------------------------------------------------------
## hearst, box
python train.py -debug 1 -dataset hearst -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 500 -lr 0.0005 -eval_method energy -loss_type contrastive -batch_size 2048 -mode box -add_denominator non -use_threshold 1 -hybrid 1 -lamda 0.0

python train.py -debug 1 -dataset hearst -train_non_basic_percent 0 -dim 10 -margin 0.001 -epoch 500 -lr 0.0005 -eval_method energy -loss_type contrastive -batch_size 1024 -mode box -add_denominator non -use_threshold 1 -hybrid 1 -lamda 0.0



## hearst, ball
python train.py -debug 1 -dataset hearst -train_non_basic_percent 0 -dim 5 -margin 0.001 -epoch 500 -lr 0.0005 -eval_method energy -loss_type contrastive -batch_size 1024 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 0.0

python train.py -debug 1 -dataset hearst -train_non_basic_percent 0 -dim 10 -margin 0.001 -epoch 500 -lr 0.001 -eval_method energy -loss_type contrastive -batch_size 1024 -mode ball -add_denominator non -use_threshold 1 -hybrid 0 -lamda 0.5 -p 2
