from scrapy.spiders import CrawlSpider,Rule
from scrapy.linkextractors import LinkExtractor
class scientificpapers(CrawlSpider):
    name='papers'
    allowed_domains=['biorxiv.org']
    start_urls=['https://www.biorxiv.org/']
    rules=(
        Rule(LinkExtractor(allow="collection") , callback="parse"),

    )
    def parse_start(self,response):
        lin=response.css('span.field-content')
        for neww in lin:
            new_url=neww.css('a::attr(href)').get()
            neww_url_page='https://www.biorxiv.org'+new_url
            yield response.follow(neww_url_page,callback=self.parse)

    def parse(self,response):
        art=response.css('.highwire-citation-biorxiv-article-pap-list-overline')
        for all in art:
            next_url = all.css('a.highwire-cite-linked-title::attr(href)').get()
            next_page_url = 'https://www.biorxiv.org' + next_url
            yield response.follow(next_page_url, callback=self.parse_inside)

        next_page = response.css('a.link-icon.link-icon-after::attr(href)').get()
        if next_page is not None:

            current_page_number = int(next_page.split('=')[-1])

            # Only proceed if the current page number is less than 5
            if current_page_number < 3:
                next_page_url = 'https://www.biorxiv.org' + next_page
                yield response.follow(next_page_url, callback=self.parse)

    def parse_inside(self, response):
        title=response.css('h1.highwire-cite-title::text').extract()[0]
        author=response.css('span.nlm-given-names::text').extract()[0:-1]
        abstract=response.css('p#p-2::text').extract()
        field=response.css('.highlight::text').extract()[0]
        yield{
            'field':field,
            'title':title,
            'author':author,
            'abstract':abstract




        }



        # for i in range(2):
        #
        #     next_page_url='https://www.biorxiv.org'+next_page
        #     yield response.follow(next_page_url, callback=self.parse)




        # yield{
        #     "title":response.css('#page-title::text').extract(),
        #     "author":response.css('.highwire-cite-authors::text').extract()
        # }