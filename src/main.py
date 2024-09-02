import subprocess
import threading
import time

def tail_file(file_path, stop_event):
    while not stop_event.is_set():
        with open(file_path, 'r') as f:
            lines = f.readlines()
            last_lines = lines[-64:]
            print('\n'.join(last_lines))
        print('-'*80)
        time.sleep(25)

def run_script_iteratively(script_path, iterations, output_file_path):
    for i in range(iterations):
        print(f"Iniciando a execução {i + 1}/{iterations}")

        # Evento para parar o tailing quando o subprocesso terminar
        stop_event = threading.Event()
        
        # Inicia a thread para ler o arquivo de saída
        tail_thread = threading.Thread(target=tail_file, args=(output_file_path, stop_event))
        tail_thread.start()
        
        with open(output_file_path, 'a') as stdout_file:
            result = subprocess.run(['python', script_path], stdout=stdout_file, stderr=subprocess.PIPE, text=True)
            
            # Indica erro caso exista
            if result.returncode != 0:
                print(result.stderr)
                print(f"Erro na execução {i + 1}!")
                break
        # Sinaliza para parar a leitura do arquivo
        stop_event.set()
        
        # Aguarda a thread terminar
        tail_thread.join()
        
        print(f"Execução {i + 1} concluída com sucesso\n")

if __name__ == "__main__":
    script_path = 'training/train.py'  # Caminho para o script a ser executado
    iterations = 16  # Número de vezes que o script será executado
    output_file_path = '../exec_log.txt'  # Caminho para o arquivo de saída
    f = open(output_file_path, 'w')
    f.close()
    run_script_iteratively(script_path, iterations, output_file_path)
