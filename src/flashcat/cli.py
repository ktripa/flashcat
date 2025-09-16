import typer
from rich import print

app = typer.Typer(help="FlashCAT: Flash-drought toolkit")
@app.command()
def info():
    print("[bold green]FlashCAT[/] is set up!")

if __name__ == "__main__":
    app()
