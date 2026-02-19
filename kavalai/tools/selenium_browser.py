import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Selenium Browser Tool")


class BrowserAction(BaseModel):
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None
    wait_for: Optional[str] = None
    timeout: int = 10


class BrowserResult(BaseModel):
    url: str
    title: str
    content: str
    screenshot: Optional[str] = None  # Base64 encoded JPEG


_driver = None


def get_driver():
    global _driver
    if _driver is None:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        service = Service(ChromeDriverManager().install())
        _driver = webdriver.Chrome(service=service, options=chrome_options)
    return _driver


@app.post("/navigate", response_model=BrowserResult)
async def navigate(action: BrowserAction):
    driver = get_driver()
    if not action.url:
        raise HTTPException(status_code=400, detail="URL is required for navigate")

    driver.get(action.url)
    if action.wait_for:
        WebDriverWait(driver, action.timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, action.wait_for))
        )

    screenshot = driver.get_screenshot_as_base64()
    return BrowserResult(
        url=driver.current_url,
        title=driver.title,
        content=driver.page_source,
        screenshot=screenshot,
    )


@app.post("/click", response_model=BrowserResult)
async def click(action: BrowserAction):
    driver = get_driver()
    if not action.selector:
        raise HTTPException(status_code=400, detail="Selector is required for click")

    element = WebDriverWait(driver, action.timeout).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, action.selector))
    )
    element.click()

    if action.wait_for:
        WebDriverWait(driver, action.timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, action.wait_for))
        )
    else:
        time.sleep(1)  # Wait a bit for potential JS execution

    screenshot = driver.get_screenshot_as_base64()
    return BrowserResult(
        url=driver.current_url,
        title=driver.title,
        content=driver.page_source,
        screenshot=screenshot,
    )


@app.post("/type", response_model=BrowserResult)
async def type_text(action: BrowserAction):
    driver = get_driver()
    if not action.selector or action.text is None:
        raise HTTPException(
            status_code=400, detail="Selector and text are required for type"
        )

    element = WebDriverWait(driver, action.timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, action.selector))
    )
    element.clear()
    element.send_keys(action.text)
    element.send_keys(Keys.ENTER)

    if action.wait_for:
        WebDriverWait(driver, action.timeout).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, action.wait_for))
        )
    else:
        time.sleep(1)

    screenshot = driver.get_screenshot_as_base64()
    return BrowserResult(
        url=driver.current_url,
        title=driver.title,
        content=driver.page_source,
        screenshot=screenshot,
    )


@app.get("/screenshot", response_model=BrowserResult)
async def screenshot():
    driver = get_driver()
    screenshot = driver.get_screenshot_as_base64()
    return BrowserResult(
        url=driver.current_url,
        title=driver.title,
        content=driver.page_source,
        screenshot=screenshot,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
