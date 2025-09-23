import sys
import os
import importlib.util

# 主函数：根据传入的应用名运行相应的gradio应用
def run_app(app_name):
    # 获取端口号参数，如果有
    port_arg = None
    if len(sys.argv) > 2:
        port_arg = sys.argv[2]
    
    # 构建应用文件路径
    app_file = os.path.join('apps', f'{app_name}.gradio.py')
    
    # 检查文件是否存在
    if not os.path.exists(app_file):
        print(f"错误：找不到应用文件 '{app_file}'")
        print(f"请确保apps目录下存在'{app_name}.gradio.py'文件")
        sys.exit(1)
    
    try:
        # 动态导入应用模块
        spec = importlib.util.spec_from_file_location(app_name, app_file)
        app_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_module)
        
        # 检查应用模块中是否有launch_app函数或main函数
        if hasattr(app_module, 'launch_app'):
            # 如果有launch_app函数，检查是否接受port参数
            import inspect
            sig = inspect.signature(app_module.launch_app)
            if 'port' in sig.parameters:
                app_module.launch_app(port=port_arg if port_arg else 7860)
            else:
                app_module.launch_app()
        elif hasattr(app_module, 'main'):
            # 如果有main函数，临时修改sys.argv以传递端口号
            original_argv = sys.argv.copy()
            if port_arg:
                sys.argv = [sys.argv[0], port_arg]  # 修改为[script_name, port]
            else:
                sys.argv = [sys.argv[0]] # 如果没有端口号，只传递脚本名
            app_module.main()
            sys.argv = original_argv  # 恢复原始参数
        else:
            print(f"警告：应用 '{app_name}' 中未找到可执行的启动函数")
            print("请确保应用模块中有 'launch_app()' 或 'main()' 函数")
            
    except Exception as e:
        print(f"运行应用 '{app_name}' 时出错：{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法：python main.py <应用名> [端口号]")
        print("例如：python main.py cwru_cls")
        print("或者：python main.py cwru_cls 8000")
        sys.exit(1)
    
    # 获取应用名并运行
    app_name = sys.argv[1]
    run_app(app_name)