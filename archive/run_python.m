im_path = "D:\Tommy\em_targeting\Jelly_fish1.png";

venv_python = "D:\Tommy\venv_310\Scripts\python";
pyenv('Version', venv_python)

command = sprintf('%s %s --path_im %s', venv_python, "D:\Tommy\em_targeting\draw_polygon_script.py", im_path);
[status, result] = system(command);
disp(result)