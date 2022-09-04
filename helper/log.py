import time
import threading
import sys

class Log():

    lock = threading.Lock()
    use_console = False
    use_files   = False
    show_time   = True
    store_str   = False
    file_path   = []
    _Log = ""

    ERROR_HEADER     = '[ERROR]'
    WARNING_HEADER   = '[WARNING]'
    INFO_HEADER      = '[Info] '
    DEBUG_HEADER     = '[Debug] '
    EXCEPTION_HEADER = '[Exception]'

    def __init__(self, **kwargs):
        pass
    
    @staticmethod
    def log_init(args):
        Log.use_console = args.console
        Log.use_files   = args.file is not None
        Log.file_path   = args.file
        Log.store_str   = args.string
        if Log.use_files:
            for path in Log.file_path:
                with open(path, 'w') as file:
                    pass

    @staticmethod
    def log(msg):
        Log.lock.acquire()
        if Log.show_time:
            msg = time.strftime("%H:%M:%S") + " " + msg
        if Log.use_console:
            print(msg)
            sys.stdout.flush()
        if Log.use_files:
            for file_path in Log.file_path:
                with open(file_path, 'a') as f:
                    f.write(msg + '\n')
        if Log.store_str:
            Log._Log += msg + '\n'
        Log.lock.release()

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
