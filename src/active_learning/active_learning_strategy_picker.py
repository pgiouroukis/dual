import importlib

def get_active_learning_class(strategy_name) -> type:
    strategy_mapping = {
        'random': 'src.active_learning.active_learning_strategy_random.ActiveLearningStrategyRandom',
        'bas': 'src.active_learning.active_learning_strategy_bas.ActiveLearningStrategyBAS',
        'idds': 'src.active_learning.active_learning_strategy_idds.ActiveLearningStrategyIDDS',
        'dual': 'src.active_learning.active_learning_strategy_dual.ActiveLearningStrategyDUAL',
        'coreset': 'src.active_learning.active_learning_strategy_coreset.ActiveLearningStrategyCoreset',
    }
    
    if strategy_name not in strategy_mapping:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    module_path, class_name = strategy_mapping[strategy_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
