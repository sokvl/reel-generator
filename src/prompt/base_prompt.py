class BasePrompt:
    def __init__(self, core: str, header: str, tail: str=""):
        self._header = header
        self._core = core
        self._tail = tail
        self.prompt = f"{self._header} {self._core} {self._tail}"

    def __str__(self):
        return self.prompt
    @property
    def core(self):
        return self._core
    
    @core.setter
    def core(self, value):
        self._core = value
        self.prompt = f"{self._header} {self._core} {self._tail}"