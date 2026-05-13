// 替换为你的 Hugging Face Space 的实际地址
const TARGET_URL = "https://你的用户名-空间名.hf.space";

async function handleRequest(request) {
  const url = new URL(request.url);
  // 构造目标 URL（保留路径和查询参数）
  const targetUrl = new URL(TARGET_URL + url.pathname + url.search);

  // 转发请求
  let response = await fetch(targetUrl.toString(), {
    method: request.method,
    headers: request.headers,
    body: request.body,
  });

  // 处理重定向：如果返回 Location 指向 hf.space，替换成自定义域名
  if (response.status >= 300 && response.status < 400) {
    let location = response.headers.get("Location");
    if (location && location.includes("hf.space")) {
      location = location.replace(/https?:\/\/[^\/]+\.hf\.space/, "https://rag.carsonng.com");
      response = new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: { ...response.headers, Location: location },
      });
    }
  }

  return response;
}

addEventListener("fetch", event => {
  event.respondWith(handleRequest(event.request));
});