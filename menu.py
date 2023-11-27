# menu.py
#
# Menu to select custom training needs.

from parapipe import run_pipeline_parallelism
from paradata import run_data_parallelism
# from paramix import run_mixed_parallelism

def main_menu():
    print("Select the parallelism mode:")
    print("1. Pipeline Parallelism")
    print("2. Data Parallelism")
    print("3. Model Parallelism")
    print("4. Mixed Parallelism")
    mode = input("Enter your choice (1/2/3): ")

    print("Select the number of nodes to use (1-10):")
    nodes = input("Enter the number of nodes: ")

    print("Choose GPU nodes:")
    print("0. Do not use GPU nodes")
    print("1. Use one GPU node (gpu1)")
    print("2. Use both GPU nodes (gpu1 and gpu2)")
    gpu_choice = input("Enter your choice (0/1/2): ")

    if mode == '1':
        run_pipeline_parallelism(nodes, gpu_choice)
    elif mode == '2':
        run_data_parallelism(nodes, gpu_choice)
    elif mode == '3':
        print("Model Parallelism is not implemented yet.")    
    elif mode == '4':
        print("Mixed Parallelism is not implemented yet.")
        #run_mixed_parallelism(nodes, gpu_choice)

if __name__ == "__main__":
    main_menu()
