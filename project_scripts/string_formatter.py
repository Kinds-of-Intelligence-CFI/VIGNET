from string import Formatter

class ExtendedFormatter(Formatter):
    """An extended string formatter with pronoun support"""

    def format_field(self, value, format_spec):
        """Handle format specifications
        This method is called for the part after the colon in {value:p1}
        """
        singular = {
            "p1": "it",
            "p2": "it",
            "p3": "was",
            "p4": "",
            "p5": "s",
            "p6": "it's",
            "p7": "is",
            "p8": "isn't"
        }
        plural = {
            "p1": "they",
            "p2": "them",
            "p3": "were",
            "p4": "s",
            "p5": "",
            "p6": "they're",
            "p7": "are",
            "p8": "aren't"
        }

        if format_spec in singular:
            if "pair" in str(value):
                return plural[format_spec]
            return singular[format_spec]
        return super().format_field(value, format_spec)

    def convert_field(self, value, conversion):
        """Handle basic conversions (!u, !l, !c)"""
        if conversion == "u":
            return str(value).upper()
        elif conversion == "l":
            return str(value).lower()
        elif conversion == "c":
            return str(value).capitalize()
        return super().convert_field(value, conversion)