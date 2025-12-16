import subprocess
import time
import os

def run_test():
    # Запуск agi_console.py
    proc = subprocess.Popen(['python', 'agi_console.py'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='cp1251', errors='replace')

    # Команды для отправки
    commands = [
        'status',
        'query Привет, AGI!',
        '',  # Пропустить rating для query
        'save',
        'load',
        'reset',
        'log',
        'exit'
    ]

    output = ""
    for cmd in commands:
        if cmd == 'exit':
            proc.stdin.write(cmd + '\n')
            proc.stdin.flush()
            break
        proc.stdin.write(cmd + '\n')
        proc.stdin.flush()
        time.sleep(1)  # Ждать обработки
        # Читать вывод
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            output += line
            if 'Введите команду:' in line:  # Ждать следующего приглашения
                break

    # Завершить процесс
    proc.wait()
    stdout, stderr = proc.communicate()

    output += stdout
    if stderr:
        output += "STDERR: " + stderr

    # Проверить файлы
    files_check = ""
    if os.path.exists('agi_model.pth'):
        files_check += "agi_model.pth существует.\n"
    else:
        files_check += "agi_model.pth НЕ существует.\n"

    if os.path.exists('agi_structure.json'):
        files_check += "agi_structure.json существует.\n"
    else:
        files_check += "agi_structure.json НЕ существует.\n"

    # Проверить логи
    if os.path.exists('output.log'):
        with open('output.log', 'r', encoding='utf-8') as f:
            logs = f.read()
        files_check += "Логи:\n" + logs[-1000:]
    else:
        files_check += "Файл логов не найден.\n"

    return output + "\n\n" + files_check

if __name__ == "__main__":
    result = run_test()
    print(result)