from dataclasses import dataclass
from string import Formatter


@dataclass
class AnnotatedSegment:
    """A tagged text segment from template formatting."""
    text: str
    source_type: str  # "template", "variable", "switch", "filler", "pov", "coinflip", "item", "activity", "number"
    source_key: str | None  # e.g. "name_1", "switch_3", None for template literals


_CATEGORY_MAP = {
    "pov": "pov",
    "switch": "switch",
    "filler": "filler",
    "coinflip": "coinflip",
    "otherside": "coinflip",
    "item": "item",
    "activity": "activity",
    "smallnumber": "number",
    "bignumber": "number",
}


def classify_variable(field_name: str) -> tuple[str, str]:
    """Categorize a variable name by its prefix.

    Returns (source_type, source_key) where source_type is one of the
    segment categories and source_key is the full field name.
    """
    prefix = field_name.split("_")[0]
    source_type = _CATEGORY_MAP.get(prefix, "variable")
    return source_type, field_name


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


class AnnotatingFormatter(ExtendedFormatter):
    """Formatter that produces AnnotatedSegment lists instead of plain strings."""

    def annotated_format(self, format_string: str, var_dict: dict) -> list[AnnotatedSegment]:
        segments: list[AnnotatedSegment] = []
        for literal_text, field_name, format_spec, conversion in self.parse(format_string):
            if literal_text:
                segments.append(AnnotatedSegment(literal_text, "template", None))
            if field_name is not None:
                obj, _ = self.get_field(field_name, (), {"vs": var_dict})
                if conversion:
                    obj = self.convert_field(obj, conversion)
                if format_spec:
                    obj = self.format_field(obj, format_spec)
                else:
                    obj = self.format_field(obj, "")

                # extract the key from "vs[name_1]" -> "name_1"
                key = field_name
                if "[" in field_name and "]" in field_name:
                    key = field_name.split("[")[1].rstrip("]")
                source_type, source_key = classify_variable(key)
                segments.append(AnnotatedSegment(str(obj), source_type, source_key))
        return segments