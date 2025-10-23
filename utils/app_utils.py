import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AppUtils:
    @staticmethod
    def get_current_examples(file_dir):
        """获取目录中的所有文件路径列表"""
        examples = []
        if os.path.exists(file_dir):
            for filename in sorted(os.listdir(file_dir)):
                examples.append(os.path.join(file_dir, filename))
        return examples

    @staticmethod
    def generate_paddlex_model_options(base_dir) -> dict:
        """生成PaddleX模型选项字典"""
        if not os.path.isdir(base_dir):
            logger.warning(f"警告: 模型根目录 '{base_dir}' 不存在。将返回空配置。")
            return {}
        
        model_collection = {}
        try:
            for model in os.listdir(base_dir):
                model_path = os.path.join(base_dir, model)
                if os.path.isdir(model_path):  # 只添加目录
                    model_collection[model] = model_path
        except OSError as e:
            logger.error(f"读取模型目录时出错: {e}")
            
        return model_collection

    @staticmethod
    def auto_config_chinese_font():
        """
        自动扫描系统中的字体，找到可用的中文字体并进行配置。
        如果找不到，则会提示。
        """
        font_keywords = [
            'Heiti', 'SimHei', 'FangSong', 'KaiTi', 'Lantinghei', '儷黑',
            'Microsoft YaHei', 'Microsoft JhengHei',
            'Noto Sans CJK', 'Source Han Sans', 'WenQuanYi', 'wqy-zenhei'
        ]

        font_paths = fm.findSystemFonts()
        target_font_path = None
        for keyword in font_keywords:
            for font_path in font_paths:
                if keyword.lower() in os.path.basename(font_path).lower():
                    target_font_path = font_path
                    break
            if target_font_path:
                break
        if target_font_path:
            try:
                font_prop = fm.FontProperties(fname=target_font_path)
                font_name = font_prop.get_name()
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                return plt
                
            except Exception as e:
                print(f"配置字体时出错: {e}")
                print("无法自动配置中文字体。")
                


class DirectoryHandler(FileSystemEventHandler):
    """文件系统事件处理器"""
    
    def __init__(self, monitor_manager_instance):
        super().__init__()
        self.monitor_manager = monitor_manager_instance
    
    def on_created(self, event):
        """文件/目录创建事件"""
        if not event.is_directory:
            logger.debug(f"检测到文件创建: {event.src_path}")
        self.monitor_manager.trigger_restart()
    
    def on_deleted(self, event):
        """文件/目录删除事件"""
        if not event.is_directory:
            logger.debug(f"检测到文件删除: {event.src_path}")
        self.monitor_manager.trigger_restart()
    
    def on_moved(self, event):
        """文件/目录移动/重命名事件"""
        if not event.is_directory:
            logger.debug(f"检测到文件移动: {event.src_path} -> {event.dest_path}")
        self.monitor_manager.trigger_restart()
    
    def on_modified(self, event):
        """文件修改事件"""
        if not event.is_directory:
            logger.debug(f"检测到文件修改: {event.src_path}")
            self.monitor_manager.trigger_restart()


class MultiDirectoryMonitor:
    """一个可以管理多个目录监控任务的类"""
    
    def __init__(self, restart_signal_file_name: str):
        self._directories_to_watch = set()
        self._observers = []
        self.restart_signal_file_name = restart_signal_file_name
        self._is_running = False

    def add_directory(self, path: str):
        """注册一个需要被监控的目录路径"""
        abs_path = os.path.abspath(path)
        if abs_path not in self._directories_to_watch:
            self._directories_to_watch.add(abs_path)
            logger.info(f"目录已注册监控: {path}")
            return True
        else:
            logger.debug(f"目录已存在于监控列表中: {path}")
            return False

    def remove_directory(self, path: str):
        """从监控列表中移除目录"""
        abs_path = os.path.abspath(path)
        if abs_path in self._directories_to_watch:
            self._directories_to_watch.remove(abs_path)
            logger.info(f"目录已从监控列表中移除: {path}")
            return True
        return False

    def trigger_restart(self):
        """触发应用重启"""
        logger.info("检测到文件变化，正在触发应用重启...")
        
        try:
            # 创建重启信号文件
            with open(self.restart_signal_file_name, "w", encoding='utf-8') as f:
                f.write("restart")
            logger.info(f"重启信号文件已创建: {self.restart_signal_file_name}")
        except IOError as e:
            logger.error(f"创建重启信号文件失败: {e}")
        
        # 停止监控
        self.stop_all(join_threads=False)
        logger.info("应用进程即将退出...")
        
        # 退出进程
        os._exit(0)

    def start_all(self):
        """为所有已注册的目录启动监控"""
        if self._is_running:
            logger.info("监控已经在运行中。")
            return False

        if not self._directories_to_watch:
            logger.warning("没有注册任何监控目录，无法启动监控。")
            return False

        # 创建事件处理器
        handler = DirectoryHandler(self)
        
        # 为每个目录创建观察者
        for path in self._directories_to_watch:
            try:
                # 确保目录存在
                os.makedirs(path, exist_ok=True)
                
                # 创建观察者
                observer = Observer()
                observer.schedule(handler, path, recursive=True)
                self._observers.append(observer)
                logger.debug(f"已为目录 {path} 创建观察者")
                
            except OSError as e:
                logger.error(f"创建监控目录 {path} 失败: {e}")
                continue
        
        # 启动所有观察者
        successful_starts = 0
        for observer in self._observers:
            try:
                observer.start()
                successful_starts += 1
            except Exception as e:
                logger.error(f"启动观察者失败: {e}")
        
        if successful_starts > 0:
            self._is_running = True
            logger.info(f"✅ 已成功启动对 {successful_starts} 个目录的监控。")
            return True
        else:
            logger.error("❌ 没有成功启动任何监控任务。")
            self.stop_all(join_threads=False)
            return False

    def stop_all(self, join_threads: bool = True):
        """停止所有监控任务"""
        if not self._observers:
            logger.debug("没有运行中的监控任务。")
            return

        # 停止所有观察者
        for observer in self._observers:
            if observer.is_alive():
                try:
                    observer.stop()
                except Exception as e:
                    logger.error(f"停止观察者时出错: {e}")

        # 等待线程结束
        if join_threads:
            for observer in self._observers:
                try:
                    observer.join(timeout=5.0)  # 设置超时时间
                    if observer.is_alive():
                        logger.warning("观察者线程未能在超时时间内结束")
                except Exception as e:
                    logger.error(f"等待观察者线程结束时出错: {e}")
        
        self._observers = []
        self._is_running = False
        logger.info("✅ 所有监控任务已停止。")

    def is_running(self):
        """检查监控是否正在运行"""
        return self._is_running

    def get_monitored_directories(self):
        """获取当前监控的目录列表"""
        return list(self._directories_to_watch)

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出，自动清理资源"""
        self.stop_all(join_threads=True)