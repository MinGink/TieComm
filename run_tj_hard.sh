#hard
python main.py --agent ac_mlp                    --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent ac_att                    --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent tiecomm_random            --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent tiecomm    --block no     --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent tiecomm    --block inter  --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent tiecomm    --block intra  --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &


python main.py --agent commnet                   --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent ic3net                    --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent tarmac                    --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &
python main.py --agent magic                     --env tj --map hard --use_multiprocessing --n_processes 8 --total_epoches 20000   --batch_size 200 &