#python main.py --agent ac_mlp --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &
#python main.py --agent ac_att --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &
#python main.py --agent tiecomm_random --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &
#python main.py --agent tiecomm_one --env tj --map easy --use_multiprocessing --n_processes 2 --epoch_size 4 --batch_size 200 &


#python main.py --agent ac_mlp --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &
#python main.py --agent ac_att --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &
#python main.py --agent tiecomm_random --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &
#python main.py --agent tiecomm_one --env mpe --map pz-mpe-large-spread-v1 --use_multiprocessing --n_processes 4 --epoch_size 5 --batch_size 200 &

#lbforaging:Foraging-10x10-3p-3f-v2

#python main.py --agent tiecomm --total_epoches 500 --env lbf --map Foraging-medium-v0 --block no --use_multiprocessing --memo aaai &
#python main.py --agent tarmac --total_epoches 500 --env lbf --map Foraging-medium-v0 --block no --use_multiprocessing --memo aaai &
#python main.py --agent tiecomm_default   --env lbf --map Foraging-easy-v0  --use_multiprocessing --memo vison3 &
#python main.py --agent tiecomm           --env lbf --map Foraging-easy-v0  --use_multiprocessing --memo vison3 &
#python main.py --agent magic            -env lbf --map Foraging-easy-v0  --use_multiprocessing &

#python main.py --agent ac_mlp   --env lbf --map Foraging-easy-v0   --time_limit 40  --total_epoches 500 --use_multiprocessing --memo env_test &
#python main.py --agent ac_att   --env lbf --map Foraging-easy-v0   --time_limit 40  --total_epoches 500 --use_multiprocessing --memo env_test &

#python main.py --agent ac_mlp   --env lbf --map Foraging-medium-v0  --time_limit 60  --total_epoches 500 --use_multiprocessing --memo env_test &
#python main.py --agent ac_att   --env lbf --map Foraging-medium-v0  --time_limit 60  --total_epoches 500 --use_multiprocessing --memo env_test &

python main.py --agent ac_mlp   --env lbf --map Foraging-hard-v0  --time_limit 80 --total_epoches 500 --use_multiprocessing --memo env_test &
python main.py --agent ac_att   --env lbf --map Foraging-hard-v0  --time_limit 80 --total_epoches 500 --use_multiprocessing --memo env_test &

#python main.py --agent tarmac --total_epoches 500 --env lbf --map Foraging-easy-v0 --block no --use_multiprocessing --memo aaai &