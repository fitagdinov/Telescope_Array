 # TASKS

1. Датасет. Считывать из файла данные только в init. В дальнейшем, может быть, надо переписать на прямое считывание из файла. (done)
2. Разобраться, что есть что в LSTM. Нам нужно h или c?( Nov11_00-14-53_cluster61.inr.ac.ru, Nov11_00-30-13_cluster61.inr.ac.ru)
3. Если как бы использовать маску в lstm, то надо считывать output[ last_det+1 ]? 
4. Считывать маски в датасете. Добавить возможность накладывать маску.
5. В декодере,немного увеличить длину последовательности. Это добавит вариабельность в данные.
6. Узнать, как в NLP генерируется предложения - что подается на вход lstm и как долго она генерирует.
7. Попробовать несколько LInear слоев после lstm.
8. Tensorboard (done)