"""Callbacks used when training and using models.
"""


class Callbacks:
    """Collection of callbacks to execute on different events.
    """

    def __init__(self):
        self.callbacks = {}

    def on(self, event: str, callback_func):
        if not event in self.callbacks:
            self.callbacks[event] = [callback_func]
        else:
            self.callbacks[event].append(callback_func)        

    def execute(self, event, **kvargs):        
        if event in self.callbacks.keys():            
            for callback in self.callbacks[event]:
                callback(**kvargs)