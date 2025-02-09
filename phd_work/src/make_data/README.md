1. filter_h5ml.py - накладываются фильтры (5 минут).
    cuts:
        - for adrons [0,1,1,0]
        - for photons [0,0,0,1]

2. merge_and_shuffle_iter.py - перемешивание событий. (Примерно 3 часа)

3. make_bundle_iter_mask.py - преобразует данные к нужному формату. (30-60 минут)

4. Нормировка - get_normalization.py (быстро) и make_normilized_iter.py (1825 секунд).