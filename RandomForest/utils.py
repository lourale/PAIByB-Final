import random

def split_dataset_3way(dataset, train_qty=600, val_qty=200):
    """
    Divide el dataset de imágenes originales en Entrenamiento, Validación y Test.
    Garantiza que no haya fuga de datos (Data Leakage) entre los grupos.
    """
    train_data = {'Normal': {}, 'Bleeding': {}, 'Ischemia': {}}
    val_data = {'Normal': {}, 'Bleeding': {}, 'Ischemia': {}}
    test_data = {'Normal': {}, 'Bleeding': {}, 'Ischemia': {}}
    
    print("-" * 50)
    print(f"SEPARACIÓN 3-WAY (Train: {train_qty} | Val: {val_qty} | Test: Resto)")
    print("-" * 50)
    
    for cls in ['Normal', 'Bleeding', 'Ischemia']:
        if cls not in dataset or dataset[cls] is None:
            continue
            
        dicoms = dataset[cls]['dicom']
        brains = dataset[cls]['brain']
        overlays = dataset[cls].get('overlay', [None] * len(dicoms))
        
        # Agrupamos para no mezclar pares
        combined = list(zip(dicoms, brains, overlays))
        
        # Mezclamos aleatoriamente con semilla fija
        random.seed(42)
        random.shuffle(combined)
        
        # Hacemos los 3 cortes exactos
        train_split = combined[:train_qty]
        val_split = combined[train_qty : train_qty + val_qty]
        test_split = combined[train_qty + val_qty:]
        
        # Función auxiliar para rearmar los diccionarios sin repetir código
        def unpack_to_dict(split_list, target_dict):
            target_dict[cls]['dicom'] = [item[0] for item in split_list]
            target_dict[cls]['brain'] = [item[1] for item in split_list]
            if cls != 'Normal':
                target_dict[cls]['overlay'] = [item[2] for item in split_list]
            else:
                target_dict[cls]['overlay'] = None

        # Desempaquetamos
        unpack_to_dict(train_split, train_data)
        unpack_to_dict(val_split, val_data)
        unpack_to_dict(test_split, test_data)
            
        print(f"[{cls}] -> Train: {len(train_split)} | Val: {len(val_split)} | Test: {len(test_split)}")

    return train_data, val_data, test_data

# --- USO ---
# train_dataset, val_dataset, test_dataset = split_dataset_3way(dataset_final, train_qty=600, val_qty=200)