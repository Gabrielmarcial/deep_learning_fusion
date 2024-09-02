import subprocess, sys

def run_script_iteratively(script_path, iterations, file_path):
    for i in range(iterations):
        print(f"Iniciando a execução {i + 1}/{iterations}")
        
        with open(file_path, 'a') as f:
            result = subprocess.run(['python', script_path], stdout=f, stderr=subprocess.PIPE, text=True)
            
            print(result.stderr, file=sys.stderr)
            
            if result.returncode != 0:
                print(f"Interrompendo por erro na execução {i + 1}!")
                break
        
        print(f"Execução {i + 1} concluída com sucesso\n")

if __name__ == "__main__":
    script_path = 'training/train.py'
    iterations = 16
    file_path = '../exec_log.txt'
    run_script_iteratively(script_path, iterations, file_path)
