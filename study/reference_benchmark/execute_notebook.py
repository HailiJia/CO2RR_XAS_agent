"""Execute the portable notebook through IPython without a socket-based kernel.

Useful in restricted environments. Every code cell actually runs; errors abort.
"""
from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = Path(__file__).resolve().parent/'Cu_XAS_revised.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            count = shell.execution_count
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            result.raise_error()
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        notebook.metadata['execution_route'] = 'IPython InteractiveShell; all code cells executed in order; no network kernel'
        nbformat.write(notebook, path)
    finally:
        os.chdir(previous)
    print('Executed all code cells:', path.name)


if __name__ == '__main__':
    main()
