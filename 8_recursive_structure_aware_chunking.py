"""Markdown ko structure aur natural boundaries ke saath chunks mein baantein."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Demo text mein headings aur paragraphs hain, jinko splitter pehle preserve karega.
markdown_text = """# Solar System

The solar system includes the Sun and everything that orbits it. Gravity holds
the system together, from the largest planets to small pieces of dust.

## The Sun

The Sun is a star at the center of the solar system. Its gravity keeps planets
and other objects in orbit. The Sun also provides most of the light and heat
that reach Earth.

## The Planets

There are eight planets. The four inner planets are Mercury, Venus, Earth, and
Mars. They are relatively small and have rocky surfaces.

The four outer planets are Jupiter, Saturn, Uranus, and Neptune. They are much
larger than the inner planets. Jupiter and Saturn are gas giants, while Uranus
and Neptune are ice giants.

### Orbits

Each planet travels around the Sun along an orbit. A planet's distance from the
Sun and its speed affect how long one orbit takes.

### Dwarf Planets

Dwarf planets orbit the Sun but have not cleared other objects from their
orbital neighborhood. Pluto is a well-known dwarf planet.
"""

# Separators ko pehle badi structure se chhoti structure ke order mein try karo.
# Separator heading ko agle section ke saath rakhta hai ("start").
SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    " ",
    "",
]


def main() -> None:
    # chunk_size aur overlap yahan characters mein count hote hain, tokens mein nahi.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=420,
        chunk_overlap=60,
        separators=SEPARATORS,
        keep_separator="start",
        add_start_index=True,
    )

    # Markdown ko Document objects mein badlo; source metadata har chunk ke saath rahega.
    documents = splitter.create_documents(
        [markdown_text],
        metadatas=[{"source": "solar_system_demo.md"}],
    )

    # Har chunk ka size, source location aur content print karo.
    print(f"Created {len(documents)} chunks:\n")
    for index, document in enumerate(documents, start=1):
        start_index = document.metadata.get("start_index", "unknown")
        print(
            f"--- Chunk {index} "
            f"({len(document.page_content)} characters, starts at {start_index}) ---"
        )
        print(f"Source: {document.metadata['source']}")
        print(document.page_content)
        print()


if __name__ == "__main__":
    main()
