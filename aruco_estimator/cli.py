
# your_package/cli.py
import click


@click.group()
def main():
    """Your CLI tool description"""
    pass

@main.command()
@click.argument('name')
@click.option('--greeting', '-g', default='Hello', help='Greeting to use')
def greet(name, greeting):
    """Greet someone."""
    click.echo(f"{greeting}, {name}!")

@main.command()
@click.argument('x', type=float)
@click.argument('y', type=float)
def add(x, y):
    """Add two numbers."""
    result = x + y
    click.echo(f"Result: {result}")

if __name__ == '__main__':
    main()