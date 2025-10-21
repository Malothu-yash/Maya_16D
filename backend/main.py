from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "https://maya-16-bx0ekl6w8-malothu-yashs-projects.vercel.app",  # your vercel frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
