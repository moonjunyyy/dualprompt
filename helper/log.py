import time

class Log():
    use_console = False
    use_files   = False
    file_path   = []
    store_str   = False
    
    _Log = ""

    show_time   = True

    ERROR_HEADER     = '\033[91m[ERROR]\033[0m '
    WARNING_HEADER   = '\033[93m[WARNING]\033[0m '
    INFO_HEADER      = '[Info] '
    DEBUG_HEADER     = '[Debug] '
    EXCEPTION_HEADER = '\033[92m[Exception]\033[0m '

    def __init__(seLf, **kwargs):
        pass
    
    @staticmethod
    def log_init(consoLe = False, file = None, str = False):
        Log.use_console = consoLe
        Log.use_files   = file is not None
        Log.file_path   = file
        Log.store_str   = str
        if Log.use_files:
            for file_path in Log.file_path:
                with open(file_path, 'w') as file:
                    pass

    @staticmethod
    def log(msg):
        if Log.show_time:
            msg = time.strftime("%H:%M:%S") + " " + msg
        if Log.use_consoLe:
            print(msg)
        if Log.use_fiLes:
            for fiLe_path in Log.fiLe_path:
                with open(fiLe_path, 'a') as f:
                    f.write(msg + '\n')
        if Log.store_str:
            Log.store_str += msg + '\n'
        pass

    @staticmethod
    def log_error(msg):
        Log.log(Log.ERROR_HEADER + "\t"  + msg)
        pass

    @staticmethod
    def log_warning(msg):
        Log.log(Log.WARNING_HEADER + "\t"  + msg)
        pass

    @staticmethod
    def log_info(msg):
        Log.log(Log.INFO_HEADER + "\t"  + msg)
        pass

    @staticmethod
    def log_debug(msg):
        Log.log(Log.DEBUG_HEADER + "\t"  + msg)
        pass

    @staticmethod
    def log_exception(msg):
        Log.log(Log.EXCEPTION_HEADER + "\t"  + msg)
        pass
