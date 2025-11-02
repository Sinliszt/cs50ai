import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    pages = list(corpus.keys())
    N = len(pages)

    links = corpus[page]
    if len(links) == 0:
        links = set(pages)

    distribution = {}

    for p in pages:
        distribution[p] = (1 - damping_factor) / N
        if p in links:
            distribution[p] += damping_factor / len(links)

    return distribution

def sample_pagerank(corpus, damping_factor, n):

    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages = list(corpus.keys())
    counts = {p: 0 for p in pages}

    current_p = random.choice(pages)

    for i in range(n):
        counts[current_p] += 1
        probability = transition_model(corpus, current_p, damping_factor)
        current_p = random.choices(
            population = list(probability.keys()),
            weights = list(probability.values()),
            k = 1
        ) [0]

    pagerank = {
        p: counts[p] / n for p in pages
    }

    return pagerank

def iterate_pagerank(corpus, damping_factor):

    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages = list(corpus.keys())
    N = len(pages)

    adjusted_corpus = {}

    for p in pages:
        if len(corpus[p]) == 0:
            adjusted_corpus[p] = set(pages)
        else:
            adjusted_corpus[p] = corpus[p]

    pagerank = {p: 1 / N for p in pages}

    while True:
        new_rank = {}

        for p in pages:
            rank = (1 - damping_factor) / N

            for q in pages:
                if p in adjusted_corpus[q]:
                    rank += damping_factor * pagerank[q] / len(adjusted_corpus[q])

            new_rank[p] = rank

        diff = max(abs(new_rank[p] - pagerank[p]) for p in pages)
        pagerank = new_rank

        if diff <= 0.001:
            break

    total = sum(pagerank.values())
    pagerank = {p: rank / total for p, rank in pagerank.items()}

    return pagerank

if __name__ == "__main__":
    main()
