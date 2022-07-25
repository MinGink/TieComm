#python main.py --agent ac_mlp --env tj --map easy --use_multiprocessing --n_processes 4 -- epoch_size 5 --batch_size 200
#python main.py --agent ac_att --env tj --map easy --use_multiprocessing --n_processes 4 -- epoch_size 5 --batch_size 200
#python main.py --agent tiecomm_random --env tj --map easy --use_multiprocessing --n_processes 4 -- epoch_size 5 --batch_size 200



python main.py --agent ac_mlp --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 -- epoch_size 5 --batch_size 200 &
python main.py --agent ac_att --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 -- epoch_size 5 --batch_size 200 &
python main.py --agent tiecomm_random --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 -- epoch_size 5 --batch_size 200 &