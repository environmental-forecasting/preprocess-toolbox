import typer

app = typer.Typer()


@app.command()
def projection():
    pass


@app.command()
def data_filter():
    pass


@app.command()
def missing_dates():
    pass


@app.command()
def missing_locations():
    pass


if __name__ == "__main__":
    app()
