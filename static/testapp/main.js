class ArxivRandomArticle {
  constructor() {
    this.apiUrl = "http://export.arxiv.org/api/query?";
    this.published = "2019-01-01";
    this.searchQuery = "all%22machine+learning%22";
  }

  // Get random article, retry if no article is found
  async getRandomArticles() {
    let article = null;
    let searchText = document.getElementById("search").value;
    let published = document.getElementById("published").value;
    let date = new Date(published);
    if (!searchText) {
      searchText = "machine+learning";
    }
    searchText = "all%22" + searchText.replace(/ /g, "+") + "%22";

    let articleIDs = [];
    let articles = [];

    // Generate random start index to simulate random article fetching
    const randomIndex = Math.floor(Math.random() * 1000);
    const url = `${this.apiUrl}search_query=${searchText}&start=${randomIndex}&max_results=256`;
    try {
      const response = await fetch(url);
      const data = await response.text();

      // Parse XML response
      const parser = new DOMParser();
      const xml = parser.parseFromString(data, "application/xml");

      // remove papers that are not published after the date
      const entries = xml.getElementsByTagName("entry");
      let articleID = 0;
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i];
        const published =
          entry.getElementsByTagName("published")[0].textContent;
        const publishedDate = new Date(published);
        if (publishedDate >= date) {
          articleID = i;
          console.log("found!", articleID, publishedDate);
          articleIDs.push(articleID);
        }
      }

      // If an entry is found, extract article data
      if (articleIDs.length > 0) {
        for (let i = 0; i < articleIDs.length; i++) {
          const entry = entries[articleIDs[i]];
          const article = {
            title: entry.getElementsByTagName("title")[0].textContent.trim(),
            authors: Array.from(entry.getElementsByTagName("author")).map(
              (author) => author.getElementsByTagName("name")[0].textContent,
            ),
            summary: entry
              .getElementsByTagName("summary")[0]
              .textContent.trim(),
            published: entry.getElementsByTagName("published")[0].textContent,
            link: entry.getElementsByTagName("link")[0].getAttribute("href"),
          };
          articles.push(article);
        }
      }
    } catch (error) {
      console.error("Error fetching article:", error);
    }

    console.log("Article found:", articles);
    return articles;
  }

  // Helper function to add a delay
  sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

/* LikeCarousel (c) 2019 Simone P.M. github.com/simonepm - Licensed MIT */
class Carousel {
  constructor(element) {
    this.board = element;
    this.articles = [];
    this.currentArticle = 0;
    this.init();
    this.push();
    this.handle();
    //    this.removeTopCard();

    this.addKeyboardListeners(); // Add this line to set up keyboard listeners
    this.currentArticle = 0;
  }

  async init() {
    this.articles = [];
    let arxiv = new ArxivRandomArticle();
    while (this.articles.length === 0) {
      this.articles = await arxiv.getRandomArticles();
    }
    //this.articles = await arxiv.getRandomArticles();
    // set the title to finished
    //this.removeTopCard();
    this.handle();
    let title = document.getElementById("title");
    title.textContent = "Start Swiping!"; // this.articles[0].title;
    //this.removeTopCard();
    return;
  }

  // Add this new method to handle keyboard events
  addKeyboardListeners() {
    document.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") {
        this.simulateSwipe("left");
      } else if (e.key === "ArrowRight") {
        this.simulateSwipe("right");
      }
    });
  }

  // Add this new method to simulate swipes
  simulateSwipe(direction) {
    const swipeEvent = {
      deltaX:
        direction === "left" ? -this.board.clientWidth : this.board.clientWidth,
      deltaY: 0,
      isFinal: true,
      direction:
        direction === "left" ? Hammer.DIRECTION_LEFT : Hammer.DIRECTION_RIGHT,
    };
    this.onPan(swipeEvent);
  }

  updateTitle() {
    let title = document.getElementById("title");
    title.textContent = this.articles[this.currentArticle].title;
  }

  removeTopCard() {
    // remove swiped card
    this.board.removeChild(this.topCard);
  }

  handle() {
    // list all cards
    this.cards = this.board.querySelectorAll(".card");
    // get top card
    this.topCard = this.cards[this.cards.length - 1];
    // get next card
    this.nextCard = this.cards[this.cards.length - 2];

    // if at least one card is present
    if (this.cards.length > 0) {
      // set default top card position and scale
      this.topCard.style.transform =
        "translateX(-50%) translateY(-50%) rotate(0deg) rotateY(0deg) scale(1)";

      // destroy previous Hammer instance, if present
      if (this.hammer) this.hammer.destroy();

      // listen for tap and pan gestures on top card
      this.hammer = new Hammer(this.topCard);
      this.hammer.add(new Hammer.Tap());
      this.hammer.add(
        new Hammer.Pan({
          position: Hammer.position_ALL,
          threshold: 0,
        }),
      );

      // pass events data to custom callbacks
      this.hammer.on("tap", (e) => {
        this.onTap(e);
      });
      this.hammer.on("pan", (e) => {
        this.onPan(e);
      });
    }
  }

  onTap(e) {
    // get finger position on top card
    let propX =
      (e.center.x - e.target.getBoundingClientRect().left) /
      e.target.clientWidth;

    // get rotation degrees around Y axis (+/- 15) based on finger position
    let rotateY = 15 * (propX < 0.05 ? -1 : 1);

    // enable transform transition
    this.topCard.style.transition = "transform 100ms ease-out";

    // apply rotation around Y axis
    this.topCard.style.transform =
      "translateX(-50%) translateY(-50%) rotate(0deg) rotateY(" +
      rotateY +
      "deg) scale(1)";

    // wait for transition end
    setTimeout(() => {
      // reset transform properties
      this.topCard.style.transform =
        "translateX(-50%) translateY(-50%) rotate(0deg) rotateY(0deg) scale(1)";

      // Get the stored article data
      let storedArticle = this.articles[this.currentArticle - 1];
      // If article data exists, open the article URL in a new tab
      if (storedArticle && storedArticle.link) {
        window.open(storedArticle.link, "_blank");
      }
    }, 100);
  }

  onPan(e) {
    if (!this.isPanning) {
      this.isPanning = true;

      // remove transition properties
      this.topCard.style.transition = null;
      if (this.nextCard) this.nextCard.style.transition = null;

      // get top card coordinates in pixels
      let style = window.getComputedStyle(this.topCard);
      let mx = style.transform.match(/^matrix\((.+)\)$/);
      this.startPosX = mx ? parseFloat(mx[1].split(", ")[4]) : 0;
      this.startPosY = mx ? parseFloat(mx[1].split(", ")[5]) : 0;

      // get top card bounds
      let bounds = this.topCard.getBoundingClientRect();

      // get finger position on top card, top (1) or bottom (-1)
      this.isDraggingFrom =
        e.center.y - bounds.top > this.topCard.clientHeight / 2 ? -1 : 1;
    }

    // get new coordinates
    let posX = e.deltaX + this.startPosX;
    let posY = e.deltaY + this.startPosY;
    // get ratio between swiped pixels and the axes
    let propX = e.deltaX / this.board.clientWidth;
    let propY = e.deltaY / this.board.clientHeight;
    // get swipe direction, left (-1) or right (1)
    let dirX = e.deltaX < 0 ? -1 : 1;
    // get degrees of rotation, between 0 and +/- 45
    let deg = this.isDraggingFrom * dirX * Math.abs(propX) * 45;
    // get scale ratio, between .95 and 1
    let scale = (95 + 5 * Math.abs(propX)) / 100;
    // move and rotate top card
    this.topCard.style.transform =
      "translateX(" +
      posX +
      "px) translateY(" +
      posY +
      "px) rotate(" +
      deg +
      "deg) rotateY(0deg) scale(1)";
    // scale up next card
    if (this.nextCard)
      this.nextCard.style.transform =
        "translateX(-50%) translateY(-50%) rotate(0deg) rotateY(0deg) scale(" +
        scale +
        ")";

    if (e.isFinal) {
      this.isPanning = false;
      let successful = false;
      // set back transition properties
      this.topCard.style.transition = "transform 200ms ease-out";
      if (this.nextCard)
        this.nextCard.style.transition = "transform 100ms linear";
      // check threshold and movement direction
      if (propX > 0.25 && e.direction == Hammer.DIRECTION_RIGHT) {
        successful = true;
        // get right border position
        posX = this.board.clientWidth;

        // Add the article title with href to a table at the bottom of the page
        var storedArticle = this.articles[this.currentArticle - 1];

        if (storedArticle && storedArticle.title && storedArticle.link) {
          let table = document.getElementById("articleTable");
          if (!table) {
            table = document.createElement("table");
            table.id = "articleTable";
            document.body.appendChild(table);
          }
          const row = table.insertRow();
          const cell = row.insertCell();
          const articleLink = document.createElement("a");
          articleLink.href = storedArticle.link;
          articleLink.textContent = storedArticle.title;
          articleLink.target = "_blank";
          cell.appendChild(articleLink);
          // add published Date
          const publishedCell = row.insertCell();
          publishedCell.textContent = storedArticle.published;
        }
      } else if (propX < -0.25 && e.direction == Hammer.DIRECTION_LEFT) {
        successful = true;
        // get left border position
        posX = -(this.board.clientWidth + this.topCard.clientWidth);
      } else if (propY < -0.25 && e.direction == Hammer.DIRECTION_UP) {
        successful = true;
        // get top border position
        posY = -(this.board.clientHeight + this.topCard.clientHeight);
      }

      if (successful) {
        // throw card in the chosen direction
        this.topCard.style.transform =
          "translateX(" +
          posX +
          "px) translateY(" +
          posY +
          "px) rotate(" +
          deg +
          "deg)";

        // wait transition end
        setTimeout(() => {
          // remove swiped card
          this.board.removeChild(this.topCard);
          // add new card
          this.push();
          // handle gestures on new top card
          this.handle();
        }, 200);
      } else {
        // reset cards position and size
        this.topCard.style.transform =
          "translateX(-50%) translateY(-50%) rotate(0deg) rotateY(0deg) scale(1)";
        if (this.nextCard)
          this.nextCard.style.transform =
            "translateX(-50%) translateY(-50%) rotate(0deg) rotateY(0deg) scale(0.95)";
      }
    }
  }

  push() {
    let card = document.createElement("div");
    card.classList.add("card");
    card.style.backgroundImage =
      "url('https://picsum.photos/320/320/?random=" +
      Math.round(Math.random() * 1000000) +
      "')";
    // add the title to the card
    let title = document.createElement("h1");
    title.setAttribute("id", "title");
    let storedArticle = this.articles[this.currentArticle]; // get the current article
    title.textContent = storedArticle
      ? storedArticle.title
      : "Looking for hot papers in your area..."; // article title or loading message
    // wait for this.articles to be fetched
    let age = document.createElement("p");
    age.textContent = storedArticle ? storedArticle.published : "";
    age.setAttribute("id", "age");

    let authors = document.createElement("p");
    authors.textContent = storedArticle ? storedArticle.authors.join(", ") : "";
    authors.setAttribute("id", "authors");
    card.appendChild(title);
    card.appendChild(age);
    card.appendChild(authors);
    this.board.insertBefore(card, this.board.firstChild);
    this.currentArticle++;
  }

  deincrement() {
    this.currentArticle--;
  }
}
