from dotenv import load_dotenv

load_dotenv()

import jinja2

jinja_env = jinja2.Environment(
    loader=jinja2.PackageLoader("jailbroken"),
    autoescape=jinja2.select_autoescape(),
    undefined=jinja2.StrictUndefined,
)
