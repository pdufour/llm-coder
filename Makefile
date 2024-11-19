run:
	static-web-server -p 3000 --cors-allow-origins "*" --cors-allow-headers "*" --cors-expose-headers "*" -d llm-coder -e false -z
