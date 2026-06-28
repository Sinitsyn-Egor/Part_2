html_content = """
<html>
<title>Data Science is Fun</title>

<body>
    <h1>Data Science is Fun</h1>
    <div id='paragraphs' class='text'>
        <p id='paragraph 0'>Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0 Paragraph 0
            Paragraph 0 Paragraph 0 Paragraph 0 </p>
        <p id='paragraph 1'>Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1 Paragraph 1
            Paragraph 1 Paragraph 1 Paragraph 1 </p>
        <p id='paragraph 2'>Here is a link to <a href='https://www.mail.ru'>Mail ru</a></p>
    </div>
    <div id='list' class='text'>
        <h2>Common Data Science Libraries</h2>
        <ul>
            <li>NumPy</li>
            <li>SciPy</li>
            <li>Pandas</li>
            <li>Scikit-Learn</li>
        </ul>
    </div>
    <div id='empty' class='empty'></div>
</body>

</html>
"""

from bs4 import BeautifulSoup as bs  # noqa: E402

soup = bs(html_content, "lxml")

print(soup.find("title"))

print(type(soup))

title = soup.find("title")
print(title)

# print(soup.body.text)

# print(soup.bode.p)


# pList = soup.body.find_all('p')
# for i, p in enumerate(pList):
#     print(p.text)

print(soup.find(id="paragraph 2"))

print(soup.find("div", class_="text"))

print(soup.find_all("div", class_="text"))

dList = soup.find_all("div", class_="text")

for div in dList:
    print(div.get("id"))


soup.find(id="paragraph 0").decompose()
soup.find(id="paragraph 1").decompose()

# print(soup)
new_p = soup.new_tag("p")
print(new_p)

new_p.string = "DDDDDDDDDDDDDDD"
print(new_p)

soup.find(id="empty").append(new_p)

print(soup)

from urllib.request import urlopen  # noqa: E402

url = "https://sklearn.org/stable/modules/generated/sklearn.decomposition.PCA.html"
html_content = urlopen(url).read()

# print(html_content[:1000])

sp = bs(html_content, "lxml")
print(sp.title.text)
