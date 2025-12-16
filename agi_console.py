# agi_console.py - Консольное приложение для Global AGI
from graph.fractalgraph import FractalGraph
from graph.graphmanipulator import GraphManipulator
from utils.text_processor import TextProcessor
import torch
import threading
import time
import logging
import os
import json

# Настройка логирования
logging.basicConfig(filename='output.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Загрузить конфигурацию из файла."""
    with open('config.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def load_qa_dataset(config):
    """Загрузить датасет Q&A из файла."""
    dataset_file = config.get('qa_dataset_file', 'qa_dataset.json')
    with open(dataset_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_dataset(qa_dataset, config):
    """Подготовка датасета для обучения."""
    # Установить VOCAB_SIZE
    vocab_size = config.get('vocab_size', 100)
    TextProcessor.set_vocab_size(vocab_size)
    input_size = config.get('network_params', {}).get('input_size', 10)

    # Инициализировать VOCAB
    for question, answer in qa_dataset:
        TextProcessor.tokenize(question)
        TextProcessor.tokenize(answer)
    # Заполнить VOCAB
    all_words = set()
    for q, a in qa_dataset:
        all_words.update(TextProcessor.tokenize(q))
        all_words.update(TextProcessor.tokenize(a))
    for word in sorted(all_words):
        if len(TextProcessor.VOCAB) < TextProcessor.VOCAB_SIZE:
            TextProcessor.VOCAB[word] = len(TextProcessor.VOCAB)

    inputs = []
    targets = []
    for question, answer in qa_dataset:
        q_embed = TextProcessor.embed_text(question, input_size)
        # Targets: one-hot для слов в ответе, суммированные
        a_logits = torch.zeros(TextProcessor.VOCAB_SIZE)
        tokens = TextProcessor.tokenize(answer)
        for token in tokens:
            if token in TextProcessor.VOCAB:
                idx = TextProcessor.VOCAB[token]
                a_logits[idx] += 1
        inputs.append(q_embed)
        targets.append(a_logits)
    return torch.stack(inputs), torch.stack(targets)

def background_training(graph: FractalGraph):
    """Фоновое обучение каждые 60 секунд."""
    while True:
        time.sleep(60)
        try:
            if graph.training_data:
                logger.info("Starting background training")
                print("Фоновое обучение...")
                graph.train_on_accumulated_data()
                print("Фоновое обучение завершено.")
                logger.info("Background training completed")
        except Exception as e:
            logger.error(f"Error in background training: {e}")
            print(f"Ошибка в фоновом обучении: {e}")

def main():
    try:
        config = load_config()
        qa_dataset = load_qa_dataset(config)
        graph = FractalGraph("agi_graph", config=config)
        manipulator = GraphManipulator(managing_object=graph)
        logger.info("AGI Console started")

        # Обучение на датасете при инициализации
        inputs, targets = prepare_dataset(qa_dataset, config)
        graph.self_train(inputs, targets)
        logger.info("Initial training completed")

        # Запуск фонового обучения
        training_thread = threading.Thread(target=background_training, args=(graph,), daemon=True)
        training_thread.start()
        logger.info("Background training thread started")

        print("Global AGI Console - Фрактальная графовая архитектура")
        print("Команды: query <запрос>, status, save, load, reset, log, train, exit")
        while True:
            try:
                command = input("Введите команду: ").strip()
                logger.info(f"Command received: {command}")
                if command.lower() == 'exit':
                    logger.info("Exiting application")
                    break
                elif command.lower() == 'status':
                    status = graph.get_status()
                    print(status)
                    logger.info("Status displayed")
                elif command.lower() == 'save':
                    graph.save_state()
                    print("Состояние сохранено.")
                    logger.info("State saved")
                elif command.lower() == 'load':
                    graph.load_state()
                    print("Состояние загружено.")
                    logger.info("State loaded")
                elif command.lower() == 'reset':
                    graph.reset()
                    print("Граф сброшен.")
                    logger.info("Graph reset")
                elif command.lower() == 'log':
                    if os.path.exists('output.log'):
                        with open('output.log', 'r', encoding='utf-8') as f:
                            logs = f.read()
                        print("Последние логи:")
                        print(logs[-1000:])  # Показать последние 1000 символов
                    else:
                        print("Файл логов не найден.")
                    logger.info("Logs displayed")
                elif command.lower() == 'train':
                    graph.train_on_accumulated_data()
                    print("Обучение на накопленных данных завершено.")
                    logger.info("Manual training completed")
                elif command.startswith('query '):
                    query = command[6:].strip()
                    if not query:
                        print("Введите запрос после 'query '")
                        continue
                    response, node, embedding, target_logits = graph.process_text_query(query)
                    print(f"Ответ: {response}")
                    logger.info(f"Query: {query}, Response: {response}")

                    # Самоадаптация: запрос оценки и обучение на основе нее
                    try:
                        rating = input("Оцените ответ (1-5) или нажмите Enter для пропуска: ").strip()
                        if rating:
                            rating = int(rating)
                            if 1 <= rating <= 5:
                                if rating >= 3:
                                    # Положительная оценка: добавить данные и обучить
                                    node.training_data.append((embedding, target_logits))
                                    graph.train_nodes_on_own_data(epochs=1)
                                    print(f"Спасибо за оценку {rating}! Вершина обучена.")
                                else:
                                    # Низкая оценка: не обучать, возможно, скорректировать
                                    print(f"Спасибо за оценку {rating}. Ответ не использован для обучения.")
                                logger.info(f"Rating received: {rating}")
                            else:
                                print("Оценка должна быть от 1 до 5.")
                        else:
                            # Без оценки: все равно обучить, так как диалог продолжается
                            node.training_data.append((embedding, target_logits))
                            graph.train_nodes_on_own_data(epochs=1)
                            logger.info("Rating skipped, trained anyway")
                    except ValueError:
                        print("Неверная оценка.")
                        logger.warning("Invalid rating input")
                else:
                    print("Неизвестная команда. Используйте: query <запрос>, status, save, load, reset, log, train, exit")
                    logger.warning(f"Unknown command: {command}")
            except Exception as e:
                print(f"Ошибка обработки команды: {e}")
                logger.error(f"Error processing command: {e}")
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        logger.error(f"Critical error: {e}")

if __name__ == "__main__":
    main()