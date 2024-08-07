<!doctype html>
<html class="">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
		<meta name="description" content="We’re on a journey to advance and democratize artificial intelligence through open source and open science." />
		<meta property="fb:app_id" content="1321688464574422" />
		<meta name="twitter:card" content="summary_large_image" />
		<meta name="twitter:site" content="@huggingface" />
		<meta property="og:title" content="app.py · tim3in/book-inventory at main" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="https://huggingface.co/spaces/tim3in/book-inventory/blob/main/app.py" />
		<meta property="og:image" content="https://cdn-thumbnails.huggingface.co/social-thumbnails/spaces/tim3in/book-inventory.png" />

		<link rel="stylesheet" href="/front/build/kube-0932549/style.css" />

		<link rel="preconnect" href="https://fonts.gstatic.com" />
		<link
			href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,200;0,300;0,400;0,600;0,700;0,900;1,200;1,300;1,400;1,600;1,700;1,900&display=swap"
			rel="stylesheet"
		/>
		<link
			href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap"
			rel="stylesheet"
		/>

		<link
			rel="preload"
			href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/katex.min.css"
			as="style"
			onload="this.onload=null;this.rel='stylesheet'"
		/>
		<noscript>
			<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/katex.min.css" />
		</noscript>

		<link rel="canonical" href="https://huggingface.co/spaces/tim3in/book-inventory/blob/main/app.py">  <!-- HEAD_svelte-1oal594_START --><style>.blob-line-num::before {
			content: attr(data-line-num);
		}
	</style><!-- HEAD_svelte-1oal594_END -->

		<title>app.py · tim3in/book-inventory at main</title>

		<script
			defer
			data-domain="huggingface.co"
			event-loggedIn="true"
			src="/js/script.pageview-props.js"
		></script>
		<script>
			window.plausible =
				window.plausible ||
				function () {
					(window.plausible.q = window.plausible.q || []).push(arguments);
				};
		</script>
		<script>
			window.hubConfig = JSON.parse(`{"features":{"signupDisabled":false},"sshGitUrl":"git@hf.co","moonHttpUrl":"https://huggingface.co","captchaApiKey":"bd5f2066-93dc-4bdd-a64b-a24646ca3859","captchaDisabledOnSignup":true,"datasetViewerPublicUrl":"https://datasets-server.huggingface.co","stripePublicKey":"pk_live_x2tdjFXBCvXo2FFmMybezpeM00J6gPCAAc","environment":"production","userAgent":"HuggingFace (production)","spacesIframeDomain":"hf.space","spacesApiUrl":"https://api.hf.space","docSearchKey":"ece5e02e57300e17d152c08056145326e90c4bff3dd07d7d1ae40cf1c8d39cb6","logoDev":{"apiUrl":"https://img.logo.dev/","apiKey":"pk_UHS2HZOeRnaSOdDp7jbd5w"}}`);
		</script>
		<script type="text/javascript" src="https://de5282c3ca0c.edge.sdk.awswaf.com/de5282c3ca0c/526cf06acb0d/challenge.js" defer></script>
	</head>
	<body class="flex flex-col min-h-screen bg-white dark:bg-gray-950 text-black ViewerBlobPage">
		<div class="flex min-h-screen flex-col">
	<div class="SVELTE_HYDRATER contents" data-target="MainHeader" data-props="{&quot;classNames&quot;:&quot;&quot;,&quot;avatarUrl&quot;:&quot;/avatars/6735bb71d96eb00e95419fab679e512c.svg&quot;,&quot;isWide&quot;:false,&quot;isZh&quot;:false,&quot;user&quot;:&quot;tim3in&quot;,&quot;unreadNotifications&quot;:0,&quot;csrf&quot;:&quot;eyJkYXRhIjp7ImV4cGlyYXRpb24iOjE3MjMxMTAwMTU0MzUsInVzZXJJZCI6IjY0NDJiYjUzN2ZlYjg2NjgxMWIyMzI4ZiJ9LCJzaWduYXR1cmUiOiIwNmY2ZmZhNWQ1Mzg5ZmYyYzYxYjljMmU0ODQwNjJhYjJhYmVjYWMwMDZlMzAzZDI5OTY1NmM3NmMwMmQxZmIwIn0=&quot;,&quot;canCreateOrg&quot;:true}"><header class="border-b border-gray-100 "><div class="w-full px-4 container flex h-16 items-center"><div class="flex flex-1 items-center"><a class="mr-5 flex flex-none items-center lg:mr-6" href="/"><img alt="Hugging Face's logo" class="w-7 md:mr-2" src="/front/assets/huggingface_logo-noborder.svg">
				<span class="hidden whitespace-nowrap text-lg font-bold md:block">Hugging Face</span></a>
			<div class="relative flex-1 lg:max-w-sm mr-2 sm:mr-4 md:mr-3 xl:mr-6"><input autocomplete="off" class="w-full dark:bg-gray-950 pl-8 form-input-alt h-9 pr-3 focus:shadow-xl " name="" placeholder="Search models, datasets, users..."   spellcheck="false" type="text" value="">
	<svg class="absolute left-2.5 text-gray-400 top-1/2 transform -translate-y-1/2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M30 28.59L22.45 21A11 11 0 1 0 21 22.45L28.59 30zM5 14a9 9 0 1 1 9 9a9 9 0 0 1-9-9z" fill="currentColor"></path></svg>
	</div>
			<div class="flex flex-none items-center justify-center p-0.5 place-self-stretch lg:hidden"><button class="relative z-40 flex h-6 w-8 items-center justify-center" type="button"><svg width="1em" height="1em" viewBox="0 0 10 10" class="text-xl" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" preserveAspectRatio="xMidYMid meet" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M1.65039 2.9999C1.65039 2.8066 1.80709 2.6499 2.00039 2.6499H8.00039C8.19369 2.6499 8.35039 2.8066 8.35039 2.9999C8.35039 3.1932 8.19369 3.3499 8.00039 3.3499H2.00039C1.80709 3.3499 1.65039 3.1932 1.65039 2.9999ZM1.65039 4.9999C1.65039 4.8066 1.80709 4.6499 2.00039 4.6499H8.00039C8.19369 4.6499 8.35039 4.8066 8.35039 4.9999C8.35039 5.1932 8.19369 5.3499 8.00039 5.3499H2.00039C1.80709 5.3499 1.65039 5.1932 1.65039 4.9999ZM2.00039 6.6499C1.80709 6.6499 1.65039 6.8066 1.65039 6.9999C1.65039 7.1932 1.80709 7.3499 2.00039 7.3499H8.00039C8.19369 7.3499 8.35039 7.1932 8.35039 6.9999C8.35039 6.8066 8.19369 6.6499 8.00039 6.6499H2.00039Z"></path></svg>
		</button>

	</div></div>
		<nav aria-label="Main" class="ml-auto hidden lg:block"><ul class="flex items-center space-x-1.5 xl:space-x-2"><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-indigo-700" href="/models"><svg class="mr-1.5 text-gray-400 group-hover:text-indigo-500" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-quaternary" d="M20.23 7.24L12 12L3.77 7.24a1.98 1.98 0 0 1 .7-.71L11 2.76c.62-.35 1.38-.35 2 0l6.53 3.77c.29.173.531.418.7.71z" opacity=".25" fill="currentColor"></path><path class="uim-tertiary" d="M12 12v9.5a2.09 2.09 0 0 1-.91-.21L4.5 17.48a2.003 2.003 0 0 1-1-1.73v-7.5a2.06 2.06 0 0 1 .27-1.01L12 12z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M20.5 8.25v7.5a2.003 2.003 0 0 1-1 1.73l-6.62 3.82c-.275.13-.576.198-.88.2V12l8.23-4.76c.175.308.268.656.27 1.01z" fill="currentColor"></path></svg>
					Models</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-red-700" href="/datasets"><svg class="mr-1.5 text-gray-400 group-hover:text-red-500" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 25 25"><ellipse cx="12.5" cy="5" fill="currentColor" fill-opacity="0.25" rx="7.5" ry="2"></ellipse><path d="M12.5 15C16.6421 15 20 14.1046 20 13V20C20 21.1046 16.6421 22 12.5 22C8.35786 22 5 21.1046 5 20V13C5 14.1046 8.35786 15 12.5 15Z" fill="currentColor" opacity="0.5"></path><path d="M12.5 7C16.6421 7 20 6.10457 20 5V11.5C20 12.6046 16.6421 13.5 12.5 13.5C8.35786 13.5 5 12.6046 5 11.5V5C5 6.10457 8.35786 7 12.5 7Z" fill="currentColor" opacity="0.5"></path><path d="M5.23628 12C5.08204 12.1598 5 12.8273 5 13C5 14.1046 8.35786 15 12.5 15C16.6421 15 20 14.1046 20 13C20 12.8273 19.918 12.1598 19.7637 12C18.9311 12.8626 15.9947 13.5 12.5 13.5C9.0053 13.5 6.06886 12.8626 5.23628 12Z" fill="currentColor"></path></svg>
					Datasets</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-blue-700" href="/spaces"><svg class="mr-1.5 text-gray-400 group-hover:text-blue-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" viewBox="0 0 25 25"><path opacity=".5" d="M6.016 14.674v4.31h4.31v-4.31h-4.31ZM14.674 14.674v4.31h4.31v-4.31h-4.31ZM6.016 6.016v4.31h4.31v-4.31h-4.31Z" fill="currentColor"></path><path opacity=".75" fill-rule="evenodd" clip-rule="evenodd" d="M3 4.914C3 3.857 3.857 3 4.914 3h6.514c.884 0 1.628.6 1.848 1.414a5.171 5.171 0 0 1 7.31 7.31c.815.22 1.414.964 1.414 1.848v6.514A1.914 1.914 0 0 1 20.086 22H4.914A1.914 1.914 0 0 1 3 20.086V4.914Zm3.016 1.102v4.31h4.31v-4.31h-4.31Zm0 12.968v-4.31h4.31v4.31h-4.31Zm8.658 0v-4.31h4.31v4.31h-4.31Zm0-10.813a2.155 2.155 0 1 1 4.31 0 2.155 2.155 0 0 1-4.31 0Z" fill="currentColor"></path><path opacity=".25" d="M16.829 6.016a2.155 2.155 0 1 0 0 4.31 2.155 2.155 0 0 0 0-4.31Z" fill="currentColor"></path></svg>
					Spaces</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-yellow-700" href="/posts"><svg class="mr-1.5 text-gray-400 group-hover:text-yellow-500 !text-yellow-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" viewBox="0 0 12 12" preserveAspectRatio="xMidYMid meet"><path fill="currentColor" fill-rule="evenodd" d="M3.73 2.4A4.25 4.25 0 1 1 6 10.26H2.17l-.13-.02a.43.43 0 0 1-.3-.43l.01-.06a.43.43 0 0 1 .12-.22l.84-.84A4.26 4.26 0 0 1 3.73 2.4Z" clip-rule="evenodd"></path></svg>
					Posts</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-yellow-700" href="/docs"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="mr-1.5 text-gray-400 group-hover:text-yellow-500" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path opacity="0.5" d="M20.9022 5.10334L10.8012 10.8791L7.76318 9.11193C8.07741 8.56791 8.5256 8.11332 9.06512 7.7914L15.9336 3.73907C17.0868 3.08811 18.5002 3.26422 19.6534 3.91519L19.3859 3.73911C19.9253 4.06087 20.5879 4.56025 20.9022 5.10334Z" fill="currentColor"></path><path d="M10.7999 10.8792V28.5483C10.2136 28.5475 9.63494 28.4139 9.10745 28.1578C8.5429 27.8312 8.074 27.3621 7.74761 26.7975C7.42122 26.2327 7.24878 25.5923 7.24756 24.9402V10.9908C7.25062 10.3319 7.42358 9.68487 7.74973 9.1123L10.7999 10.8792Z" fill="currentColor" fill-opacity="0.75"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M21.3368 10.8499V6.918C21.3331 6.25959 21.16 5.61234 20.8346 5.03949L10.7971 10.8727L10.8046 10.874L21.3368 10.8499Z" fill="currentColor"></path><path opacity="0.5" d="M21.7937 10.8488L10.7825 10.8741V28.5486L21.7937 28.5234C23.3344 28.5234 24.5835 27.2743 24.5835 25.7335V13.6387C24.5835 12.0979 23.4365 11.1233 21.7937 10.8488Z" fill="currentColor"></path></svg>
					Docs</a>
			</li>
		<li class="max-2xl:hidden"><div class="relative ">
	<button class="px-2 py-0.5 group hover:text-green-700 dark:hover:text-gray-400 flex items-center " type="button">
		<svg class="mr-1.5 text-gray-400 group-hover:text-green-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-tertiary" d="M19 6H5a3 3 0 0 0-3 3v2.72L8.837 14h6.326L22 11.72V9a3 3 0 0 0-3-3z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M10 6V5h4v1h2V5a2.002 2.002 0 0 0-2-2h-4a2.002 2.002 0 0 0-2 2v1h2zm-1.163 8L2 11.72V18a3.003 3.003 0 0 0 3 3h14a3.003 3.003 0 0 0 3-3v-6.28L15.163 14H8.837z" fill="currentColor"></path></svg>
			Solutions
		</button>
	
	
	</div></li>
		<li><a class="group flex items-center px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-400" href="/pricing">Pricing
			</a></li>

		<li><div class="relative group">
	<button class="px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-600 flex items-center " type="button">
		<svg class=" text-gray-500 w-5 group-hover:text-gray-400 dark:text-gray-300 dark:group-hover:text-gray-400" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" viewBox="0 0 32 18" preserveAspectRatio="xMidYMid meet"><path fill-rule="evenodd" clip-rule="evenodd" d="M14.4504 3.30221C14.4504 2.836 14.8284 2.45807 15.2946 2.45807H28.4933C28.9595 2.45807 29.3374 2.836 29.3374 3.30221C29.3374 3.76842 28.9595 4.14635 28.4933 4.14635H15.2946C14.8284 4.14635 14.4504 3.76842 14.4504 3.30221Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M14.4504 9.00002C14.4504 8.53382 14.8284 8.15588 15.2946 8.15588H28.4933C28.9595 8.15588 29.3374 8.53382 29.3374 9.00002C29.3374 9.46623 28.9595 9.84417 28.4933 9.84417H15.2946C14.8284 9.84417 14.4504 9.46623 14.4504 9.00002Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M14.4504 14.6978C14.4504 14.2316 14.8284 13.8537 15.2946 13.8537H28.4933C28.9595 13.8537 29.3374 14.2316 29.3374 14.6978C29.3374 15.164 28.9595 15.542 28.4933 15.542H15.2946C14.8284 15.542 14.4504 15.164 14.4504 14.6978Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M1.94549 6.87377C2.27514 6.54411 2.80962 6.54411 3.13928 6.87377L6.23458 9.96907L9.32988 6.87377C9.65954 6.54411 10.194 6.54411 10.5237 6.87377C10.8533 7.20343 10.8533 7.73791 10.5237 8.06756L6.23458 12.3567L1.94549 8.06756C1.61583 7.73791 1.61583 7.20343 1.94549 6.87377Z" fill="currentColor"></path></svg>
			
		</button>
	
	
	</div></li>
		<li><hr class="h-5 w-0.5 border-none bg-gray-100 dark:bg-gray-800"></li>
		<li><form action="/logout" method="POST" class="hidden"><input type="hidden" name="csrf" value="eyJkYXRhIjp7ImV4cGlyYXRpb24iOjE3MjMxMTAwMTU0MzUsInVzZXJJZCI6IjY0NDJiYjUzN2ZlYjg2NjgxMWIyMzI4ZiJ9LCJzaWduYXR1cmUiOiIwNmY2ZmZhNWQ1Mzg5ZmYyYzYxYjljMmU0ODQwNjJhYjJhYmVjYWMwMDZlMzAzZDI5OTY1NmM3NmMwMmQxZmIwIn0="></form>
<div class="relative ml-2 w-[1.38rem] h-[1.38rem] ">
	<button class="ml-auto rounded-full ring-2 group ring-indigo-400 focus:ring-blue-500 hover:ring-offset-1 focus:ring-offset-1 focus:outline-none outline-none dark:ring-offset-gray-950 " type="button">
		
		<div class="relative"><img alt="" class="h-[1.38rem] w-[1.38rem] overflow-hidden rounded-full" src="/avatars/6735bb71d96eb00e95419fab679e512c.svg" crossorigin="anonymous">
			</div>
	
		</button>
	
	
	</div></li></ul></nav></div></header></div>
	
	
	
	<div class="SVELTE_HYDRATER contents" data-target="SSOBanner" data-props="{&quot;organizations&quot;:[]}"></div>
	

	<main class="flex flex-1 flex-col"><div class="SVELTE_HYDRATER contents" data-target="SpaceHeader" data-props="{&quot;activeTab&quot;:&quot;files&quot;,&quot;authLight&quot;:{&quot;csrfToken&quot;:&quot;eyJkYXRhIjp7ImV4cGlyYXRpb24iOjE3MjMxMTAwMTU0MzUsInVzZXJJZCI6IjY0NDJiYjUzN2ZlYjg2NjgxMWIyMzI4ZiJ9LCJzaWduYXR1cmUiOiIwNmY2ZmZhNWQ1Mzg5ZmYyYzYxYjljMmU0ODQwNjJhYjJhYmVjYWMwMDZlMzAzZDI5OTY1NmM3NmMwMmQxZmIwIn0=&quot;,&quot;hasHfLevelAccess&quot;:false,&quot;u&quot;:{&quot;avatarUrl&quot;:&quot;/avatars/6735bb71d96eb00e95419fab679e512c.svg&quot;,&quot;isPro&quot;:false,&quot;orgs&quot;:[],&quot;user&quot;:&quot;tim3in&quot;,&quot;canPost&quot;:false,&quot;canHaveBilling&quot;:true,&quot;canCreateOrg&quot;:true,&quot;theme&quot;:&quot;light&quot;,&quot;notifications&quot;:{&quot;org_suggestions&quot;:false,&quot;daily_papers_digest&quot;:true}}},&quot;author&quot;:{&quot;avatarUrl&quot;:&quot;/avatars/6735bb71d96eb00e95419fab679e512c.svg&quot;,&quot;fullname&quot;:&quot;Timothy Malche&quot;,&quot;name&quot;:&quot;tim3in&quot;,&quot;type&quot;:&quot;user&quot;,&quot;isPro&quot;:false,&quot;isHf&quot;:false,&quot;isMod&quot;:false},&quot;canDisable&quot;:false,&quot;canReadRepoSettings&quot;:true,&quot;canWriteRepoSettings&quot;:true,&quot;discussionsStats&quot;:{&quot;closed&quot;:0,&quot;open&quot;:0,&quot;total&quot;:0},&quot;query&quot;:{},&quot;space&quot;:{&quot;author&quot;:&quot;tim3in&quot;,&quot;colorFrom&quot;:&quot;purple&quot;,&quot;colorTo&quot;:&quot;green&quot;,&quot;cardData&quot;:{&quot;title&quot;:&quot;Book Inventory&quot;,&quot;emoji&quot;:&quot;🏢&quot;,&quot;colorFrom&quot;:&quot;purple&quot;,&quot;colorTo&quot;:&quot;green&quot;,&quot;sdk&quot;:&quot;gradio&quot;,&quot;sdk_version&quot;:&quot;4.40.0&quot;,&quot;app_file&quot;:&quot;app.py&quot;,&quot;pinned&quot;:false,&quot;license&quot;:&quot;cc-by-4.0&quot;},&quot;createdAt&quot;:&quot;2024-08-06T11:10:09.000Z&quot;,&quot;emoji&quot;:&quot;🏢&quot;,&quot;discussionsDisabled&quot;:false,&quot;duplicationDisabled&quot;:false,&quot;id&quot;:&quot;tim3in/book-inventory&quot;,&quot;isLikedByUser&quot;:false,&quot;isMutedByUser&quot;:false,&quot;isWatchedByUser&quot;:true,&quot;lastModified&quot;:&quot;2024-08-07T08:30:48.000Z&quot;,&quot;likes&quot;:0,&quot;pinned&quot;:false,&quot;private&quot;:false,&quot;gated&quot;:false,&quot;repoType&quot;:&quot;space&quot;,&quot;subdomain&quot;:&quot;tim3in-book-inventory&quot;,&quot;sdk&quot;:&quot;gradio&quot;,&quot;sdkVersion&quot;:&quot;4.40.0&quot;,&quot;title&quot;:&quot;Book Inventory&quot;,&quot;runtime&quot;:{&quot;stage&quot;:&quot;RUNNING&quot;,&quot;hardware&quot;:{&quot;current&quot;:&quot;cpu-basic&quot;,&quot;requested&quot;:&quot;cpu-basic&quot;},&quot;storage&quot;:null,&quot;gcTimeout&quot;:172800,&quot;replicas&quot;:{&quot;current&quot;:1,&quot;requested&quot;:1},&quot;devMode&quot;:false,&quot;domains&quot;:[{&quot;domain&quot;:&quot;tim3in-book-inventory.hf.space&quot;,&quot;isCustom&quot;:false,&quot;stage&quot;:&quot;READY&quot;}],&quot;sha&quot;:&quot;4655f5dfb6de5d87aa07b30c85577b7df062dfd9&quot;},&quot;iframe&quot;:{&quot;embedSrc&quot;:&quot;https://tim3in-book-inventory.hf.space&quot;,&quot;src&quot;:&quot;https://tim3in-book-inventory.hf.space&quot;},&quot;secrets&quot;:[{&quot;key&quot;:&quot;AI71_API_KEY&quot;,&quot;updatedAt&quot;:&quot;2024-08-06T12:21:53.431Z&quot;}],&quot;variables&quot;:[],&quot;sse&quot;:{&quot;url&quot;:&quot;https://api.hf.space/v1/tim3in/book-inventory&quot;,&quot;jwt&quot;:&quot;eyJhbGciOiJFZERTQSJ9.eyJyZWFkIjp0cnVlLCJwZXJtaXNzaW9ucyI6eyJyZXBvLmNvbnRlbnQucmVhZCI6dHJ1ZX0sIm9uQmVoYWxmT2YiOnsia2luZCI6InVzZXIiLCJfaWQiOiI2NDQyYmI1MzdmZWI4NjY4MTFiMjMyOGYiLCJ1c2VyIjoidGltM2luIn0sImlhdCI6MTcyMzAyMzYxNSwic3ViIjoiL3NwYWNlcy90aW0zaW4vYm9vay1pbnZlbnRvcnkiLCJleHAiOjE3MjMxMTAwMTUsImlzcyI6Imh0dHBzOi8vaHVnZ2luZ2ZhY2UuY28ifQ.MLlkKf4sks77m72Vw_S6-38rmmMR6_eo6AITlLvIfkLE9DgzExmc-wNvjAXc5PmAiHyuTWtGT-Fz_7MgoCYaAA&quot;},&quot;dockerJwt&quot;:&quot;eyJhbGciOiJFZERTQSJ9.eyJyZWFkIjp0cnVlLCJwZXJtaXNzaW9ucyI6eyJyZXBvLmNvbnRlbnQucmVhZCI6dHJ1ZX0sIm9uQmVoYWxmT2YiOnsia2luZCI6InVzZXIiLCJfaWQiOiI2NDQyYmI1MzdmZWI4NjY4MTFiMjMyOGYiLCJ1c2VyIjoidGltM2luIn0sImlhdCI6MTcyMzAyMzYxNSwic3ViIjoiL3NwYWNlcy90aW0zaW4vYm9vay1pbnZlbnRvcnkiLCJleHAiOjE3MjMxMTAwMTUsImlzcyI6Imh0dHBzOi8vaHVnZ2luZ2ZhY2UuY28ifQ.MLlkKf4sks77m72Vw_S6-38rmmMR6_eo6AITlLvIfkLE9DgzExmc-wNvjAXc5PmAiHyuTWtGT-Fz_7MgoCYaAA&quot;,&quot;linkedModels&quot;:[{&quot;author&quot;:&quot;microsoft&quot;,&quot;authorData&quot;:{&quot;avatarUrl&quot;:&quot;https://cdn-avatars.huggingface.co/v1/production/uploads/1583646260758-5e64858c87403103f9f1055d.png&quot;,&quot;fullname&quot;:&quot;Microsoft&quot;,&quot;name&quot;:&quot;microsoft&quot;,&quot;type&quot;:&quot;org&quot;,&quot;isHf&quot;:false,&quot;isMod&quot;:false,&quot;isEnterprise&quot;:false},&quot;downloads&quot;:230754,&quot;gated&quot;:false,&quot;id&quot;:&quot;microsoft/Florence-2-large&quot;,&quot;inference&quot;:&quot;custom-code&quot;,&quot;lastModified&quot;:&quot;2024-07-20T00:13:01.000Z&quot;,&quot;likes&quot;:948,&quot;pipeline_tag&quot;:&quot;image-text-to-text&quot;,&quot;private&quot;:false,&quot;repoType&quot;:&quot;model&quot;,&quot;isLikedByUser&quot;:false},{&quot;author&quot;:&quot;tiiuae&quot;,&quot;authorData&quot;:{&quot;avatarUrl&quot;:&quot;https://cdn-avatars.huggingface.co/v1/production/uploads/61a8d1aac664736898ffc84f/AT6cAB5ZNwCcqFMal71WD.jpeg&quot;,&quot;fullname&quot;:&quot;Technology Innovation Institute&quot;,&quot;name&quot;:&quot;tiiuae&quot;,&quot;type&quot;:&quot;org&quot;,&quot;isHf&quot;:false,&quot;isMod&quot;:false,&quot;isEnterprise&quot;:true},&quot;downloads&quot;:1063,&quot;gated&quot;:&quot;auto&quot;,&quot;id&quot;:&quot;tiiuae/falcon-180B-chat&quot;,&quot;inference&quot;:&quot;explicitoptout&quot;,&quot;lastModified&quot;:&quot;2023-11-07T14:38:47.000Z&quot;,&quot;likes&quot;:539,&quot;pipeline_tag&quot;:&quot;text-generation&quot;,&quot;private&quot;:false,&quot;repoType&quot;:&quot;model&quot;,&quot;isLikedByUser&quot;:false}],&quot;linkedDatasets&quot;:[],&quot;linkedCollections&quot;:[],&quot;sha&quot;:&quot;4655f5dfb6de5d87aa07b30c85577b7df062dfd9&quot;},&quot;u&quot;:{&quot;avatarUrl&quot;:&quot;/avatars/6735bb71d96eb00e95419fab679e512c.svg&quot;,&quot;isPro&quot;:false,&quot;fullname&quot;:&quot;Timothy Malche&quot;,&quot;user&quot;:&quot;tim3in&quot;,&quot;orgs&quot;:[],&quot;signup&quot;:{&quot;github&quot;:&quot;tim3in&quot;,&quot;twitter&quot;:&quot;tim3in&quot;},&quot;isHf&quot;:false,&quot;isMod&quot;:false,&quot;type&quot;:&quot;user&quot;,&quot;canPay&quot;:false,&quot;spacesAvailableFlavors&quot;:[&quot;cpu-basic&quot;,&quot;zero-a10g&quot;,&quot;cpu-upgrade&quot;,&quot;t4-small&quot;,&quot;t4-medium&quot;,&quot;l4x1&quot;,&quot;l4x4&quot;,&quot;a10g-small&quot;,&quot;a10g-large&quot;,&quot;a10g-largex2&quot;,&quot;a10g-largex4&quot;,&quot;a100-large&quot;,&quot;h100&quot;,&quot;v5e-1x1&quot;,&quot;v5e-2x2&quot;,&quot;v5e-2x4&quot;],&quot;canPost&quot;:false}}">

<header class="from-gray-50-to-white border-b border-gray-100 bg-gradient-to-t via-white dark:via-gray-950 pt-4 xl:pt-0"><div class="container relative flex flex-col xl:flex-row"><h1 class="flex flex-wrap items-center leading-tight gap-y-1 text-lg xl:flex-none"><a href="/spaces" class="group flex items-center"><svg class="mr-1 text-gray-400" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M7.80914 18.7462V24.1907H13.2536V18.7462H7.80914Z" fill="#FF3270"></path><path d="M18.7458 18.7462V24.1907H24.1903V18.7462H18.7458Z" fill="#861FFF"></path><path d="M7.80914 7.80982V13.2543H13.2536V7.80982H7.80914Z" fill="#097EFF"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M4 6.41775C4 5.08246 5.08246 4 6.41775 4H14.6457C15.7626 4 16.7026 4.75724 16.9802 5.78629C18.1505 4.67902 19.7302 4 21.4685 4C25.0758 4 28.0003 6.92436 28.0003 10.5317C28.0003 12.27 27.3212 13.8497 26.2139 15.02C27.243 15.2977 28.0003 16.2376 28.0003 17.3545V25.5824C28.0003 26.9177 26.9177 28.0003 25.5824 28.0003H17.0635H14.9367H6.41775C5.08246 28.0003 4 26.9177 4 25.5824V15.1587V14.9367V6.41775ZM7.80952 7.80952V13.254H13.254V7.80952H7.80952ZM7.80952 24.1907V18.7462H13.254V24.1907H7.80952ZM18.7462 24.1907V18.7462H24.1907V24.1907H18.7462ZM18.7462 10.5317C18.7462 9.0283 19.9651 7.80952 21.4685 7.80952C22.9719 7.80952 24.1907 9.0283 24.1907 10.5317C24.1907 12.0352 22.9719 13.254 21.4685 13.254C19.9651 13.254 18.7462 12.0352 18.7462 10.5317Z" fill="black"></path><path d="M21.4681 7.80982C19.9647 7.80982 18.7458 9.02861 18.7458 10.5321C18.7458 12.0355 19.9647 13.2543 21.4681 13.2543C22.9715 13.2543 24.1903 12.0355 24.1903 10.5321C24.1903 9.02861 22.9715 7.80982 21.4681 7.80982Z" fill="#FFD702"></path></svg>
					<span class="mr-2.5 font-semibold text-gray-400 group-hover:text-gray-500">Spaces:</span></a>
			<div class="group flex flex-none items-center"><div class="relative mr-1.5 flex items-center">

			<img alt="" class="w-3.5 h-3.5 rounded-full  flex-none" src="/avatars/6735bb71d96eb00e95419fab679e512c.svg" crossorigin="anonymous"></div>
		<a href="/tim3in" class="text-gray-400 hover:text-blue-600">tim3in</a>
		<div class="mx-0.5 text-gray-300">/</div></div>

<div class="max-w-full xl:flex xl:min-w-0 xl:flex-nowrap xl:items-center xl:gap-x-1"><a class="break-words font-mono font-semibold hover:text-blue-600 text-[1.07rem] xl:truncate" href="/spaces/tim3in/book-inventory">book-inventory</a>
	<button class="relative text-xs mr-3  inline-flex cursor-pointer items-center text-sm focus:outline-none  mx-0.5   text-gray-600 " title="Copy space name to clipboard" type="button"><svg class="" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>
	
	</button></div>
			<div class="inline-flex items-center overflow-hidden whitespace-nowrap rounded-md border bg-white text-sm leading-none text-gray-500  mr-2"><button class="relative flex items-center overflow-hidden from-red-50 to-transparent dark:from-red-900 px-1.5 py-1 hover:bg-gradient-to-t focus:outline-none"  title="Like"><svg class="left-1.5 absolute" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" fill="currentColor"><path d="M22.45,6a5.47,5.47,0,0,1,3.91,1.64,5.7,5.7,0,0,1,0,8L16,26.13,5.64,15.64a5.7,5.7,0,0,1,0-8,5.48,5.48,0,0,1,7.82,0L16,10.24l2.53-2.58A5.44,5.44,0,0,1,22.45,6m0-2a7.47,7.47,0,0,0-5.34,2.24L16,7.36,14.89,6.24a7.49,7.49,0,0,0-10.68,0,7.72,7.72,0,0,0,0,10.82L16,29,27.79,17.06a7.72,7.72,0,0,0,0-10.82A7.49,7.49,0,0,0,22.45,4Z"></path></svg>

		
		<span class="ml-4 pl-0.5 ">like</span></button>
	<button class="flex items-center border-l px-1.5 py-1 text-gray-400 hover:bg-gray-50 focus:bg-gray-100 focus:outline-none dark:hover:bg-gray-900 dark:focus:bg-gray-800" title="See users who liked this repository">0</button></div>


			



<span class="inline-block "><span class="contents"><div class="inline-flex cursor-pointer select-none items-center overflow-hidden font-mono text-xs flex-shrink-0 mr-2 rounded-lg border leading-none dark:bg-gray-900
					border-green-100 
					text-green-700 dark:text-green-500"><div class="inline-flex items-center px-2 py-[0.32rem] dark:bg-gray-900  border-green-100 bg-green-50 hover:bg-green-100/70 hover:text-green-800 dark:hover:text-green-400">
					<div class="ml-0.5 mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green-500"></div>
		Running
		</div>
	</div></span>

	</span>

	<button class="mr-2 inline-flex items-center overflow-hidden whitespace-nowrap rounded-md border bg-white px-1.5 py-1 text-sm leading-none text-gray-500 hover:bg-gray-50 focus:bg-gray-100 focus:outline-none dark:hover:bg-gray-900 dark:focus:bg-gray-800"><svg class="2xl:mr-1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path fill="currentColor" d="M4 6h18v2H4zm0 6h18v2H4zm0 6h12v2H4zm17 0l7 5l-7 5V18z"></path></svg><span class="hidden 2xl:block">Logs</span></button>


			
			

<div class="sm:hidden"><div class="relative ">
	<button class="btn px-1 py-1 text-sm translate-y-0 " type="button">
		
			<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="p-px" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><circle cx="16" cy="7" r="3" fill="currentColor"></circle><circle cx="16" cy="16" r="3" fill="currentColor"></circle><circle cx="16" cy="25" r="3" fill="currentColor"></circle></svg>
			<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" class="absolute right-[-0.25rem] bottom-[-0.25rem]  rounded-sm bg-gray-50 p-px text-[0.85rem] text-gray-500 dark:bg-gray-925" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 12 12"><path fill="currentColor" d="M7.975 3.489a.438.438 0 0 1 0 .618L4.262 7.82a.416.416 0 0 1-.307.126.427.427 0 0 1-.311-.126.438.438 0 0 1 0-.618L7.357 3.49a.438.438 0 0 1 .618 0ZM6.427 8.132 4.88 9.675a2.17 2.17 0 0 1-3.09 0 2.188 2.188 0 0 1 0-3.09l1.542-1.548a.437.437 0 0 0-.618-.619L1.166 5.966a3.063 3.063 0 0 0 4.332 4.332L7.046 8.75a.438.438 0 0 0-.619-.618Zm4.026-7.121a3.063 3.063 0 0 0-4.332 0L4.573 2.559a.438.438 0 0 0 .618.618L6.74 1.635a2.171 2.171 0 0 1 3.09 0 2.188 2.188 0 0 1 0 3.09L8.287 6.273a.432.432 0 0 0 0 .618.421.421 0 0 0 .475.097.438.438 0 0 0 .143-.097l1.548-1.548a3.068 3.068 0 0 0 0-4.332Z"></path></svg>
		
		</button>
	
	
	</div></div></h1>
		

		<div class="flex flex-col-reverse gap-x-2 sm:flex-row sm:items-center sm:justify-between xl:ml-auto"><div class="-mb-px flex h-12 items-center overflow-x-auto overflow-y-hidden sm:h-[3.25rem]"><a class="tab-alternate " href="/spaces/tim3in/book-inventory"><svg class="mr-1.5 text-gray-400 flex-none" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-quaternary" d="M20.23 7.24L12 12L3.77 7.24a1.98 1.98 0 0 1 .7-.71L11 2.76c.62-.35 1.38-.35 2 0l6.53 3.77c.29.173.531.418.7.71z" opacity=".25" fill="currentColor"></path><path class="uim-tertiary" d="M12 12v9.5a2.09 2.09 0 0 1-.91-.21L4.5 17.48a2.003 2.003 0 0 1-1-1.73v-7.5a2.06 2.06 0 0 1 .27-1.01L12 12z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M20.5 8.25v7.5a2.003 2.003 0 0 1-1 1.73l-6.62 3.82c-.275.13-.576.198-.88.2V12l8.23-4.76c.175.308.268.656.27 1.01z" fill="currentColor"></path></svg>
			App
			
			
		</a><a class="tab-alternate active" href="/spaces/tim3in/book-inventory/tree/main"><svg class="mr-1.5 text-gray-400 flex-none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-tertiary" d="M21 19h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2zm0-4h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2zm0-8h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2zm0 4h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M9 19a1 1 0 0 1-1-1V6a1 1 0 0 1 2 0v12a1 1 0 0 1-1 1zm-6-4.333a1 1 0 0 1-.64-1.769L3.438 12l-1.078-.898a1 1 0 0 1 1.28-1.538l2 1.667a1 1 0 0 1 0 1.538l-2 1.667a.999.999 0 0 1-.64.231z" fill="currentColor"></path></svg>
			<span class="xl:hidden">Files</span>
				<span class="hidden xl:inline">Files</span>
			
			
		</a><a class="tab-alternate " href="/spaces/tim3in/book-inventory/discussions"><svg class="mr-1.5 text-gray-400 flex-none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M20.6081 3C21.7684 3 22.8053 3.49196 23.5284 4.38415C23.9756 4.93678 24.4428 5.82749 24.4808 7.16133C24.9674 7.01707 25.4353 6.93643 25.8725 6.93643C26.9833 6.93643 27.9865 7.37587 28.696 8.17411C29.6075 9.19872 30.0124 10.4579 29.8361 11.7177C29.7523 12.3177 29.5581 12.8555 29.2678 13.3534C29.8798 13.8646 30.3306 14.5763 30.5485 15.4322C30.719 16.1032 30.8939 17.5006 29.9808 18.9403C30.0389 19.0342 30.0934 19.1319 30.1442 19.2318C30.6932 20.3074 30.7283 21.5229 30.2439 22.6548C29.5093 24.3704 27.6841 25.7219 24.1397 27.1727C21.9347 28.0753 19.9174 28.6523 19.8994 28.6575C16.9842 29.4379 14.3477 29.8345 12.0653 29.8345C7.87017 29.8345 4.8668 28.508 3.13831 25.8921C0.356375 21.6797 0.754104 17.8269 4.35369 14.1131C6.34591 12.058 7.67023 9.02782 7.94613 8.36275C8.50224 6.39343 9.97271 4.20438 12.4172 4.20438H12.4179C12.6236 4.20438 12.8314 4.2214 13.0364 4.25468C14.107 4.42854 15.0428 5.06476 15.7115 6.02205C16.4331 5.09583 17.134 4.359 17.7682 3.94323C18.7242 3.31737 19.6794 3 20.6081 3ZM20.6081 5.95917C20.2427 5.95917 19.7963 6.1197 19.3039 6.44225C17.7754 7.44319 14.8258 12.6772 13.7458 14.7131C13.3839 15.3952 12.7655 15.6837 12.2086 15.6837C11.1036 15.6837 10.2408 14.5497 12.1076 13.1085C14.9146 10.9402 13.9299 7.39584 12.5898 7.1776C12.5311 7.16799 12.4731 7.16355 12.4172 7.16355C11.1989 7.16355 10.6615 9.33114 10.6615 9.33114C10.6615 9.33114 9.0863 13.4148 6.38031 16.206C3.67434 18.998 3.5346 21.2388 5.50675 24.2246C6.85185 26.2606 9.42666 26.8753 12.0653 26.8753C14.8021 26.8753 17.6077 26.2139 19.1799 25.793C19.2574 25.7723 28.8193 22.984 27.6081 20.6107C27.4046 20.212 27.0693 20.0522 26.6471 20.0522C24.9416 20.0522 21.8393 22.6726 20.5057 22.6726C20.2076 22.6726 19.9976 22.5416 19.9116 22.222C19.3433 20.1173 28.552 19.2325 27.7758 16.1839C27.639 15.6445 27.2677 15.4256 26.746 15.4263C24.4923 15.4263 19.4358 19.5181 18.3759 19.5181C18.2949 19.5181 18.2368 19.4937 18.2053 19.4419C17.6743 18.557 17.9653 17.9394 21.7082 15.6009C25.4511 13.2617 28.0783 11.8545 26.5841 10.1752C26.4121 9.98141 26.1684 9.8956 25.8725 9.8956C23.6001 9.89634 18.2311 14.9403 18.2311 14.9403C18.2311 14.9403 16.7821 16.496 15.9057 16.496C15.7043 16.496 15.533 16.4139 15.4169 16.2112C14.7956 15.1296 21.1879 10.1286 21.5484 8.06535C21.7928 6.66715 21.3771 5.95917 20.6081 5.95917Z" fill="#FF9D00"></path><path d="M5.50686 24.2246C3.53472 21.2387 3.67446 18.9979 6.38043 16.206C9.08641 13.4147 10.6615 9.33111 10.6615 9.33111C10.6615 9.33111 11.2499 6.95933 12.59 7.17757C13.93 7.39581 14.9139 10.9401 12.1069 13.1084C9.29997 15.276 12.6659 16.7489 13.7459 14.713C14.8258 12.6772 17.7747 7.44316 19.304 6.44221C20.8326 5.44128 21.9089 6.00204 21.5484 8.06532C21.188 10.1286 14.795 15.1295 15.4171 16.2118C16.0391 17.2934 18.2312 14.9402 18.2312 14.9402C18.2312 14.9402 25.0907 8.49588 26.5842 10.1752C28.0776 11.8545 25.4512 13.2616 21.7082 15.6008C17.9646 17.9393 17.6744 18.557 18.2054 19.4418C18.7372 20.3266 26.9998 13.1351 27.7759 16.1838C28.5513 19.2324 19.3434 20.1173 19.9117 22.2219C20.48 24.3274 26.3979 18.2382 27.6082 20.6107C28.8193 22.9839 19.2574 25.7722 19.18 25.7929C16.0914 26.62 8.24723 28.3726 5.50686 24.2246Z" fill="#FFD21E"></path></svg>
			Community
			
			
		</a><a class="tab-alternate " href="/spaces/tim3in/book-inventory/settings"><svg class="opacity-50 dark:opacity-70 mr-1.5 text-gray-400 flex-none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 25 25" fill="currentColor"><path d="M13.0101 3C13.7157 3.0078 14.4184 3.09062 15.1077 3.24652C15.2543 3.2797 15.3871 3.35848 15.4874 3.47186C15.5877 3.58523 15.6506 3.72754 15.6672 3.8789L15.8306 5.36678C15.8537 5.57655 15.925 5.77791 16.0389 5.95464C16.1527 6.13137 16.3059 6.27854 16.4861 6.38432C16.6663 6.4901 16.8685 6.55153 17.0764 6.56367C17.2843 6.57581 17.4921 6.53831 17.6831 6.4542L19.0299 5.85495C19.1667 5.79391 19.3187 5.77743 19.4651 5.8078C19.6115 5.83818 19.745 5.9139 19.847 6.02449C20.8199 7.07789 21.5443 8.3412 21.9658 9.71937C22.01 9.8642 22.0087 10.0194 21.962 10.1634C21.9153 10.3074 21.8256 10.4332 21.7053 10.5232L20.5113 11.4158C20.3434 11.5408 20.2069 11.7041 20.1128 11.8925C20.0187 12.0809 19.9697 12.2891 19.9697 12.5003C19.9697 12.7114 20.0187 12.9196 20.1128 13.108C20.2069 13.2964 20.3434 13.4597 20.5113 13.5848L21.7062 14.4763C21.8269 14.5663 21.917 14.6922 21.9638 14.8364C22.0107 14.9806 22.0121 15.1361 21.9677 15.2812C21.546 16.6593 20.8216 17.9225 19.849 18.976C19.7471 19.0864 19.6141 19.162 19.4681 19.1926C19.3221 19.2231 19.1704 19.207 19.0338 19.1466L17.6812 18.5454C17.4904 18.4606 17.2827 18.4225 17.0748 18.4343C16.8668 18.446 16.6645 18.5072 16.4842 18.6129C16.3039 18.7185 16.1508 18.8658 16.037 19.0426C15.9233 19.2195 15.8523 19.421 15.8297 19.6308L15.6672 21.1177C15.6508 21.2674 15.5892 21.4084 15.4908 21.5212C15.3923 21.6341 15.2619 21.7133 15.1173 21.7482C13.7249 22.0839 12.2742 22.0839 10.8817 21.7482C10.7371 21.7133 10.6067 21.6341 10.5083 21.5212C10.4098 21.4084 10.3482 21.2674 10.3318 21.1177L10.1703 19.6328C10.1468 19.4235 10.0751 19.2227 9.96107 19.0465C9.84703 18.8704 9.69381 18.7238 9.51373 18.6186C9.33364 18.5134 9.13172 18.4525 8.92419 18.4408C8.71666 18.4291 8.50931 18.4669 8.31882 18.5512L6.9672 19.1514C6.83048 19.2121 6.67854 19.2283 6.53235 19.1978C6.38616 19.1672 6.25292 19.0915 6.15103 18.9809C5.17789 17.9263 4.45346 16.6616 4.03227 15.2821C3.98795 15.1371 3.98931 14.9816 4.03617 14.8374C4.08304 14.6931 4.17306 14.5673 4.29375 14.4773L5.48868 13.5848C5.65676 13.4599 5.79345 13.2966 5.88768 13.1082C5.9819 12.9198 6.031 12.7115 6.031 12.5003C6.031 12.289 5.9819 12.0808 5.88768 11.8923C5.79345 11.7039 5.65676 11.5407 5.48868 11.4158L4.29375 10.5252C4.17324 10.4351 4.0834 10.3092 4.03671 10.1649C3.99003 10.0207 3.98881 9.8653 4.03323 9.72034C4.45479 8.34219 5.17922 7.07889 6.15199 6.02547C6.25407 5.91487 6.38753 5.83915 6.53391 5.80878C6.6803 5.77841 6.83238 5.79488 6.96912 5.85593L8.31498 6.45517C8.5063 6.53923 8.71441 6.57664 8.92258 6.56439C9.13075 6.55214 9.33319 6.49057 9.51363 6.38462C9.69406 6.27868 9.84747 6.13132 9.96152 5.95438C10.0756 5.77744 10.1471 5.57585 10.1703 5.36581L10.3338 3.8789C10.3503 3.72724 10.4132 3.58462 10.5137 3.47103C10.6142 3.35745 10.7473 3.2786 10.8942 3.24555C11.5835 3.09062 12.2881 3.00877 13.0101 3ZM12.9986 9.57711C12.2337 9.57711 11.5001 9.88508 10.9593 10.4333C10.4184 10.9815 10.1146 11.725 10.1146 12.5003C10.1146 13.2755 10.4184 14.0191 10.9593 14.5672C11.5001 15.1154 12.2337 15.4234 12.9986 15.4234C13.7634 15.4234 14.497 15.1154 15.0378 14.5672C15.5787 14.0191 15.8825 13.2755 15.8825 12.5003C15.8825 11.725 15.5787 10.9815 15.0378 10.4333C14.497 9.88508 13.7634 9.57711 12.9986 9.57711Z"></path></svg>
			Settings
			
			
		</a>
	</div>
		
			

<div class="hidden sm:block mt-2 lg:mt-0"><div class="relative ">
	<button class="btn px-1 py-1 text-base translate-y-px " type="button">
		
			<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="p-0.5" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><circle cx="16" cy="7" r="3" fill="currentColor"></circle><circle cx="16" cy="16" r="3" fill="currentColor"></circle><circle cx="16" cy="25" r="3" fill="currentColor"></circle></svg>
			<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" class="absolute right-[-0.18rem] bottom-[-0.18rem]  rounded-sm bg-gray-50 p-px text-[0.85rem] text-gray-500 dark:bg-gray-925" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 12 12"><path fill="currentColor" d="M7.975 3.489a.438.438 0 0 1 0 .618L4.262 7.82a.416.416 0 0 1-.307.126.427.427 0 0 1-.311-.126.438.438 0 0 1 0-.618L7.357 3.49a.438.438 0 0 1 .618 0ZM6.427 8.132 4.88 9.675a2.17 2.17 0 0 1-3.09 0 2.188 2.188 0 0 1 0-3.09l1.542-1.548a.437.437 0 0 0-.618-.619L1.166 5.966a3.063 3.063 0 0 0 4.332 4.332L7.046 8.75a.438.438 0 0 0-.619-.618Zm4.026-7.121a3.063 3.063 0 0 0-4.332 0L4.573 2.559a.438.438 0 0 0 .618.618L6.74 1.635a2.171 2.171 0 0 1 3.09 0 2.188 2.188 0 0 1 0 3.09L8.287 6.273a.432.432 0 0 0 0 .618.421.421 0 0 0 .475.097.438.438 0 0 0 .143-.097l1.548-1.548a3.068 3.068 0 0 0 0-4.332Z"></path></svg>
		
		</button>
	
	
	</div></div>
		</div></div></header>

























	



</div>
	
<div class="container relative flex flex-col md:grid md:space-y-0 w-full md:grid-cols-12  space-y-4 md:gap-6 mb-16"><section class="pt-8 border-gray-100 col-span-full"><header class="flex flex-wrap items-center justify-start pb-2 md:justify-end lg:flex-nowrap"><div class="relative mr-4 flex min-w-0 basis-auto flex-wrap items-center md:flex-grow md:basis-full lg:basis-auto lg:flex-nowrap"><div class="SVELTE_HYDRATER contents" data-target="BranchSelector" data-props="{&quot;path&quot;:&quot;app.py&quot;,&quot;repoName&quot;:&quot;tim3in/book-inventory&quot;,&quot;repoType&quot;:&quot;space&quot;,&quot;rev&quot;:&quot;main&quot;,&quot;refs&quot;:{&quot;branches&quot;:[{&quot;name&quot;:&quot;main&quot;,&quot;ref&quot;:&quot;refs/heads/main&quot;,&quot;targetCommit&quot;:&quot;4655f5dfb6de5d87aa07b30c85577b7df062dfd9&quot;}],&quot;tags&quot;:[],&quot;converts&quot;:[]},&quot;view&quot;:&quot;blob&quot;}"><div class="relative mr-4 mb-2">
	<button class="text-sm md:text-base btn w-full cursor-pointer text-sm" type="button">
		<svg class="mr-1.5 text-gray-700 dark:text-gray-400" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24" style="transform: rotate(360deg);"><path d="M13 14c-3.36 0-4.46 1.35-4.82 2.24C9.25 16.7 10 17.76 10 19a3 3 0 0 1-3 3a3 3 0 0 1-3-3c0-1.31.83-2.42 2-2.83V7.83A2.99 2.99 0 0 1 4 5a3 3 0 0 1 3-3a3 3 0 0 1 3 3c0 1.31-.83 2.42-2 2.83v5.29c.88-.65 2.16-1.12 4-1.12c2.67 0 3.56-1.34 3.85-2.23A3.006 3.006 0 0 1 14 7a3 3 0 0 1 3-3a3 3 0 0 1 3 3c0 1.34-.88 2.5-2.09 2.86C17.65 11.29 16.68 14 13 14m-6 4a1 1 0 0 0-1 1a1 1 0 0 0 1 1a1 1 0 0 0 1-1a1 1 0 0 0-1-1M7 4a1 1 0 0 0-1 1a1 1 0 0 0 1 1a1 1 0 0 0 1-1a1 1 0 0 0-1-1m10 2a1 1 0 0 0-1 1a1 1 0 0 0 1 1a1 1 0 0 0 1-1a1 1 0 0 0-1-1z" fill="currentColor"></path></svg>
			main
		<svg class="-mr-1 text-gray-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path d="M16.293 9.293L12 13.586L7.707 9.293l-1.414 1.414L12 16.414l5.707-5.707z" fill="currentColor"></path></svg></button>
	
	
	</div></div>
		<div class="relative mb-2 flex flex-wrap items-center"><a class="truncate text-gray-800 hover:underline" href="/spaces/tim3in/book-inventory/tree/main">book-inventory</a>
			<span class="mx-1 text-gray-300">/</span>
				<span class="dark:text-gray-300">app.py</span>
				<div class="SVELTE_HYDRATER contents" data-target="CopyButton" data-props="{&quot;value&quot;:&quot;app.py&quot;,&quot;classNames&quot;:&quot;text-xs ml-2&quot;,&quot;title&quot;:&quot;Copy path&quot;}"><button class="relative text-xs ml-2 inline-flex cursor-pointer items-center text-sm focus:outline-none  mx-0.5   text-gray-600 " title="Copy path" type="button"><svg class="" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>
	
	</button></div></div></div>

	
	</header>
			<div class="SVELTE_HYDRATER contents" data-target="LastCommit" data-props="{&quot;commitLast&quot;:{&quot;date&quot;:&quot;2024-08-07T08:30:48.000Z&quot;,&quot;verified&quot;:&quot;verified&quot;,&quot;subject&quot;:&quot;Update app.py&quot;,&quot;authors&quot;:[{&quot;_id&quot;:&quot;6442bb537feb866811b2328f&quot;,&quot;avatar&quot;:&quot;/avatars/6735bb71d96eb00e95419fab679e512c.svg&quot;,&quot;isHf&quot;:false,&quot;user&quot;:&quot;tim3in&quot;}],&quot;commit&quot;:{&quot;id&quot;:&quot;4655f5dfb6de5d87aa07b30c85577b7df062dfd9&quot;,&quot;parentIds&quot;:[&quot;3562d490d880be81d22ab9d6089efe8503f15511&quot;]},&quot;title&quot;:&quot;Update app.py&quot;},&quot;repo&quot;:{&quot;name&quot;:&quot;tim3in/book-inventory&quot;,&quot;type&quot;:&quot;space&quot;}}"><div class="from-gray-100-to-white flex items-baseline rounded-t-lg border border-b-0 bg-gradient-to-t px-3 py-2 dark:border-gray-800"><img class="mr-2.5 mt-0.5 h-4 w-4 self-center rounded-full" alt="tim3in's picture" src="/avatars/6735bb71d96eb00e95419fab679e512c.svg">
			<div class="mr-5 flex flex-none items-center truncate"><a class="hover:underline" href="/tim3in">tim3in
					</a>
				
			</div>
		<div class="mr-4 truncate font-mono text-sm text-gray-500 hover:prose-a:underline"><!-- HTML_TAG_START -->Update app.py<!-- HTML_TAG_END --></div>
		<a class="rounded border bg-gray-50 px-1.5 text-sm hover:underline dark:border-gray-800 dark:bg-gray-900" href="/spaces/tim3in/book-inventory/commit/4655f5dfb6de5d87aa07b30c85577b7df062dfd9">4655f5d</a>
		<span class="mx-2 text-green-500 dark:text-green-600 px-1.5 border-green-100 dark:border-green-800 rounded-full border text-xs uppercase" title="This commit is signed and the signature is verified">verified</span>
		<time class="ml-auto hidden flex-none truncate pl-2 text-gray-500 dark:text-gray-400 lg:block" datetime="2024-08-07T08:30:48" title="Wed, 07 Aug 2024 08:30:48 GMT">about 1 hour ago</time></div></div>
			<div class="flex flex-wrap items-center border px-3 py-1.5 text-sm text-gray-800 dark:border-gray-800 dark:bg-gray-900">
				<a class="my-1 mr-4 flex items-center hover:underline " href="/spaces/tim3in/book-inventory/raw/main/app.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M31 16l-7 7l-1.41-1.41L28.17 16l-5.58-5.59L24 9l7 7z" fill="currentColor"></path><path d="M1 16l7-7l1.41 1.41L3.83 16l5.58 5.59L8 23l-7-7z" fill="currentColor"></path><path d="M12.419 25.484L17.639 6l1.932.518L14.35 26z" fill="currentColor"></path></svg>
							raw
						</a><div class="SVELTE_HYDRATER contents" data-target="CopyButton" data-props="{&quot;value&quot;:&quot;https://huggingface.co/spaces/tim3in/book-inventory/resolve/main/app.py&quot;,&quot;style&quot;:&quot;blank&quot;,&quot;label&quot;:&quot;Copy download link&quot;,&quot;classNames&quot;:&quot;my-1 mr-4 flex items-center no-underline hover:underline&quot;}"><button class="relative my-1 mr-4 flex items-center no-underline hover:underline       " title="Copy download link" type="button"><svg class="" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>
	<span class="ml-1.5 ">Copy download link</span>
	</button></div><a class="my-1 mr-4 flex items-center hover:underline " href="/spaces/tim3in/book-inventory/commits/main/app.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M16 4C9.383 4 4 9.383 4 16s5.383 12 12 12s12-5.383 12-12S22.617 4 16 4zm0 2c5.535 0 10 4.465 10 10s-4.465 10-10 10S6 21.535 6 16S10.465 6 16 6zm-1 2v9h7v-2h-5V8z" fill="currentColor"></path></svg>
							history
						</a><a class="my-1 mr-4 flex items-center hover:underline " href="/spaces/tim3in/book-inventory/blame/main/app.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2zm0 26a12 12 0 1 1 12-12a12 12 0 0 1-12 12z" fill="currentColor"></path><path d="M11.5 11a2.5 2.5 0 1 0 2.5 2.5a2.48 2.48 0 0 0-2.5-2.5z" fill="currentColor"></path><path d="M20.5 11a2.5 2.5 0 1 0 2.5 2.5a2.48 2.48 0 0 0-2.5-2.5z" fill="currentColor"></path></svg>
							blame
						</a><a class="my-1 mr-4 flex items-center hover:underline " href="/spaces/tim3in/book-inventory/edit/main/app.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M2 26h28v2H2z" fill="currentColor"></path><path d="M25.4 9c.8-.8.8-2 0-2.8l-3.6-3.6c-.8-.8-2-.8-2.8 0l-15 15V24h6.4l15-15zm-5-5L24 7.6l-3 3L17.4 7l3-3zM6 22v-3.6l10-10l3.6 3.6l-10 10H6z" fill="currentColor"></path></svg>
							edit
						</a><a class="my-1 mr-4 flex items-center hover:underline " href="/spaces/tim3in/book-inventory/delete/main/app.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M12 12h2v12h-2z" fill="currentColor"></path><path d="M18 12h2v12h-2z" fill="currentColor"></path><path d="M4 6v2h2v20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8h2V6zm4 22V8h16v20z" fill="currentColor"></path><path d="M12 2h8v2h-8z" fill="currentColor"></path></svg>
							delete
						</a>
				<div class="mr-4 flex items-center text-gray-400"><svg class="text-gray-300 text-sm mr-1.5 -translate-y-px" width="1em" height="1em" viewBox="0 0 22 28" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M15.3634 10.3639C15.8486 10.8491 15.8486 11.6357 15.3634 12.1209L10.9292 16.5551C10.6058 16.8785 10.0814 16.8785 9.7579 16.5551L7.03051 13.8277C6.54532 13.3425 6.54532 12.5558 7.03051 12.0707C7.51569 11.5855 8.30234 11.5855 8.78752 12.0707L9.7579 13.041C10.0814 13.3645 10.6058 13.3645 10.9292 13.041L13.6064 10.3639C14.0916 9.8787 14.8782 9.8787 15.3634 10.3639Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M10.6666 27.12C4.93329 25.28 0 19.2267 0 12.7867V6.52001C0 5.40001 0.693334 4.41334 1.73333 4.01334L9.73333 1.01334C10.3333 0.786673 11 0.786673 11.6 1.02667L19.6 4.02667C20.1083 4.21658 20.5465 4.55701 20.8562 5.00252C21.1659 5.44803 21.3324 5.97742 21.3333 6.52001V12.7867C21.3333 19.24 16.4 25.28 10.6666 27.12Z" fill="currentColor" fill-opacity="0.22"></path><path d="M10.0845 1.94967L10.0867 1.94881C10.4587 1.8083 10.8666 1.81036 11.2286 1.95515L11.2387 1.95919L11.2489 1.963L19.2489 4.963L19.25 4.96342C19.5677 5.08211 19.8416 5.29488 20.0351 5.57333C20.2285 5.85151 20.3326 6.18203 20.3333 6.52082C20.3333 6.52113 20.3333 6.52144 20.3333 6.52176L20.3333 12.7867C20.3333 18.6535 15.8922 24.2319 10.6666 26.0652C5.44153 24.2316 1 18.6409 1 12.7867V6.52001C1 5.82357 1.42893 5.20343 2.08883 4.94803L10.0845 1.94967Z" stroke="currentColor" stroke-opacity="0.30" stroke-width="2"></path></svg>

							No virus
						</div>
				
				<div class="flex items-center gap-x-3 dark:text-gray-300 sm:ml-auto"><div class="SVELTE_HYDRATER contents" data-target="LineWrapButton" data-props="{&quot;classNames&quot;:&quot;text-xs&quot;,&quot;lineSelectorClass&quot;:&quot;blob-line&quot;}">

<button class="text-xs" type="button" title="Toggle Line Wrap"><svg class="opacity-50" width="1em" height="1em" viewBox="0 0 12 11" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0.75 1.25H11.25M0.75 5H9C9.75 5 11.25 5.375 11.25 6.875C11.25 8.375 9.99975 8.75 9.375 8.75H6M6 8.75L7.5 7.25M6 8.75L7.5 10.25M0.75 8.75H3.75" stroke="currentColor" stroke-width="1.125" stroke-linecap="round" stroke-linejoin="round"></path></svg></button></div>
					2.82 kB</div></div>

			<div class="relative min-h-[100px] rounded-b-lg border border-t-0 leading-tight dark:border-gray-800 dark:bg-gray-925">
				<div class="py-3"><div class="SVELTE_HYDRATER contents" data-target="BlobContent" data-props="{&quot;lines&quot;:[&quot;<span class=\&quot;hljs-comment\&quot;># Import necessary libraries</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>import</span> gradio <span class=\&quot;hljs-keyword\&quot;>as</span> gr&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> PIL <span class=\&quot;hljs-keyword\&quot;>import</span> Image&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> transformers <span class=\&quot;hljs-keyword\&quot;>import</span> AutoProcessor, AutoModelForCausalLM&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> ai71 <span class=\&quot;hljs-keyword\&quot;>import</span> AI71&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>import</span> os&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>import</span> json&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>import</span> subprocess&quot;,&quot;subprocess.run(<span class=\&quot;hljs-string\&quot;>&amp;#x27;pip install flash-attn --no-build-isolation&amp;#x27;</span>, env={<span class=\&quot;hljs-string\&quot;>&amp;#x27;FLASH_ATTENTION_SKIP_CUDA_BUILD&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;quot;TRUE&amp;quot;</span>}, shell=<span class=\&quot;hljs-literal\&quot;>True</span>)&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Load the Florence-2 model and processor</span>&quot;,&quot;model = AutoModelForCausalLM.from_pretrained(<span class=\&quot;hljs-string\&quot;>&amp;quot;microsoft/Florence-2-large&amp;quot;</span>, trust_remote_code=<span class=\&quot;hljs-literal\&quot;>True</span>)&quot;,&quot;processor = AutoProcessor.from_pretrained(<span class=\&quot;hljs-string\&quot;>&amp;quot;microsoft/Florence-2-large&amp;quot;</span>, trust_remote_code=<span class=\&quot;hljs-literal\&quot;>True</span>)&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Function to run Florence-2 inference</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>run_inference</span>(<span class=\&quot;hljs-params\&quot;>image, prompt=<span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;lt;OCR&amp;gt;&amp;quot;</span></span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>try</span>:&quot;,&quot;        inputs = processor(text=prompt, images=image, return_tensors=<span class=\&quot;hljs-string\&quot;>&amp;quot;pt&amp;quot;</span>)&quot;,&quot;&quot;,&quot;        generated_ids = model.generate(&quot;,&quot;            input_ids=inputs[<span class=\&quot;hljs-string\&quot;>&amp;quot;input_ids&amp;quot;</span>],&quot;,&quot;            pixel_values=inputs[<span class=\&quot;hljs-string\&quot;>&amp;quot;pixel_values&amp;quot;</span>],&quot;,&quot;            max_new_tokens=<span class=\&quot;hljs-number\&quot;>1024</span>,&quot;,&quot;            num_beams=<span class=\&quot;hljs-number\&quot;>3</span>,&quot;,&quot;            do_sample=<span class=\&quot;hljs-literal\&quot;>False</span>&quot;,&quot;        )&quot;,&quot;        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=<span class=\&quot;hljs-literal\&quot;>False</span>)[<span class=\&quot;hljs-number\&quot;>0</span>]&quot;,&quot;        parsed_answer = processor.post_process_generation(generated_text, task=<span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;lt;OCR&amp;gt;&amp;quot;</span>, image_size=(image.width, image.height))&quot;,&quot;&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> parsed_answer&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>except</span> Exception <span class=\&quot;hljs-keyword\&quot;>as</span> e:&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> <span class=\&quot;hljs-string\&quot;>f&amp;quot;Error in run_inference: <span class=\&quot;hljs-subst\&quot;>{<span class=\&quot;hljs-built_in\&quot;>str</span>(e)}</span>&amp;quot;</span>&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Function to extract book details from text using AI71</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>extract_details</span>(<span class=\&quot;hljs-params\&quot;>text</span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>try</span>:&quot;,&quot;       &quot;,&quot;&quot;,&quot;        AI71_api_key = os.getenv(<span class=\&quot;hljs-string\&quot;>&amp;quot;AI71_API_KEY&amp;quot;</span>)&quot;,&quot;        &quot;,&quot;        client = AI71(AI71_api_key)&quot;,&quot;&quot;,&quot;        <span class=\&quot;hljs-comment\&quot;># Simple invocation</span>&quot;,&quot;        result = client.chat.completions.create(&quot;,&quot;            model=<span class=\&quot;hljs-string\&quot;>&amp;quot;tiiuae/falcon-180B-chat&amp;quot;</span>,&quot;,&quot;            messages=[&quot;,&quot;                {<span class=\&quot;hljs-string\&quot;>&amp;quot;role&amp;quot;</span>: <span class=\&quot;hljs-string\&quot;>&amp;quot;user&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;content&amp;quot;</span>: <span class=\&quot;hljs-string\&quot;>f&amp;quot;Extract book title, description, author names, publisher, and suggest genre from the following text: <span class=\&quot;hljs-subst\&quot;>{text}</span> and generate JSON output.&amp;quot;</span>},&quot;,&quot;            ],&quot;,&quot;        )&quot;,&quot;&quot;,&quot;        content = result.choices[<span class=\&quot;hljs-number\&quot;>0</span>].message.content&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> content&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>except</span> Exception <span class=\&quot;hljs-keyword\&quot;>as</span> e:&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> <span class=\&quot;hljs-string\&quot;>f&amp;quot;Error in extracting details: <span class=\&quot;hljs-subst\&quot;>{<span class=\&quot;hljs-built_in\&quot;>str</span>(e)}</span>&amp;quot;</span>&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Function to handle the combined pipeline</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>combined_pipeline</span>(<span class=\&quot;hljs-params\&quot;>image</span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>try</span>:&quot;,&quot;        <span class=\&quot;hljs-comment\&quot;># Run Florence-2 inference on the uploaded image</span>&quot;,&quot;        extracted_text = run_inference(image)&quot;,&quot;&quot;,&quot;        <span class=\&quot;hljs-comment\&quot;># Extract details from the text using Falcon</span>&quot;,&quot;        book_details = extract_details(extracted_text)&quot;,&quot;&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> book_details&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>except</span> Exception <span class=\&quot;hljs-keyword\&quot;>as</span> e:&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> <span class=\&quot;hljs-string\&quot;>f&amp;quot;Error in combined_pipeline: <span class=\&quot;hljs-subst\&quot;>{<span class=\&quot;hljs-built_in\&quot;>str</span>(e)}</span>&amp;quot;</span>&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Create the Gradio interface</span>&quot;,&quot;interface = gr.Interface(&quot;,&quot;    fn=combined_pipeline,&quot;,&quot;    inputs=gr.Image(<span class=\&quot;hljs-built_in\&quot;>type</span>=<span class=\&quot;hljs-string\&quot;>&amp;quot;pil&amp;quot;</span>, mirror_webcam=<span class=\&quot;hljs-literal\&quot;>False</span>),&quot;,&quot;    outputs=gr.Textbox(),&quot;,&quot;    title=<span class=\&quot;hljs-string\&quot;>&amp;quot;Automate Book Inventory using AI&amp;quot;</span>,&quot;,&quot;    description=<span class=\&quot;hljs-string\&quot;>&amp;quot;Upload an image of a BOOK cover to run Florence-2 inference to extract OCR and understand and prepare structured book information using Falcon.&amp;quot;</span>&quot;,&quot;)&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Launch the Gradio app</span>&quot;,&quot;interface.launch(share=<span class=\&quot;hljs-literal\&quot;>True</span>)&quot;],&quot;lineSelectorClass&quot;:&quot;blob-line&quot;,&quot;context&quot;:{&quot;repo&quot;:{&quot;name&quot;:&quot;tim3in/book-inventory&quot;,&quot;type&quot;:&quot;space&quot;},&quot;revision&quot;:&quot;4655f5dfb6de5d87aa07b30c85577b7df062dfd9&quot;,&quot;path&quot;:&quot;app.py&quot;}}">

<div class="relative text-sm"><div class="overflow-x-auto"><table class="min-w-full border-collapse font-mono"><tbody><tr class="" id="L1">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="1"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Import necessary libraries</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L2">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="2"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">import</span> gradio <span class="hljs-keyword">as</span> gr<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L3">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="3"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L4">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="4"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> transformers <span class="hljs-keyword">import</span> AutoProcessor, AutoModelForCausalLM<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L5">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="5"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> ai71 <span class="hljs-keyword">import</span> AI71<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L6">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="6"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">import</span> os<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L7">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="7"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">import</span> json<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L8">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="8"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L9">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="9"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">import</span> subprocess<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L10">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="10"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->subprocess.run(<span class="hljs-string">&#x27;pip install flash-attn --no-build-isolation&#x27;</span>, env={<span class="hljs-string">&#x27;FLASH_ATTENTION_SKIP_CUDA_BUILD&#x27;</span>: <span class="hljs-string">&quot;TRUE&quot;</span>}, shell=<span class="hljs-literal">True</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L11">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="11"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L12">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="12"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Load the Florence-2 model and processor</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L13">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="13"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->model = AutoModelForCausalLM.from_pretrained(<span class="hljs-string">&quot;microsoft/Florence-2-large&quot;</span>, trust_remote_code=<span class="hljs-literal">True</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L14">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="14"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->processor = AutoProcessor.from_pretrained(<span class="hljs-string">&quot;microsoft/Florence-2-large&quot;</span>, trust_remote_code=<span class="hljs-literal">True</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L15">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="15"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L16">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="16"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Function to run Florence-2 inference</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L17">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="17"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">run_inference</span>(<span class="hljs-params">image, prompt=<span class="hljs-string">&quot;&lt;OCR&gt;&quot;</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L18">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="18"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">try</span>:<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L19">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="19"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        inputs = processor(text=prompt, images=image, return_tensors=<span class="hljs-string">&quot;pt&quot;</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L20">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="20"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L21">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="21"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        generated_ids = model.generate(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L22">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="22"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            input_ids=inputs[<span class="hljs-string">&quot;input_ids&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L23">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="23"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            pixel_values=inputs[<span class="hljs-string">&quot;pixel_values&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L24">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="24"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            max_new_tokens=<span class="hljs-number">1024</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L25">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="25"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            num_beams=<span class="hljs-number">3</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L26">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="26"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            do_sample=<span class="hljs-literal">False</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L27">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="27"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L28">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="28"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=<span class="hljs-literal">False</span>)[<span class="hljs-number">0</span>]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L29">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="29"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        parsed_answer = processor.post_process_generation(generated_text, task=<span class="hljs-string">&quot;&lt;OCR&gt;&quot;</span>, image_size=(image.width, image.height))<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L30">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="30"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L31">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="31"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> parsed_answer<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L32">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="32"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">except</span> Exception <span class="hljs-keyword">as</span> e:<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L33">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="33"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> <span class="hljs-string">f&quot;Error in run_inference: <span class="hljs-subst">{<span class="hljs-built_in">str</span>(e)}</span>&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L34">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="34"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L35">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="35"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Function to extract book details from text using AI71</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L36">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="36"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">extract_details</span>(<span class="hljs-params">text</span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L37">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="37"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">try</span>:<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L38">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="38"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->       <!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L39">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="39"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L40">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="40"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        AI71_api_key = os.getenv(<span class="hljs-string">&quot;AI71_API_KEY&quot;</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L41">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="41"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L42">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="42"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        client = AI71(AI71_api_key)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L43">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="43"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L44">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="44"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-comment"># Simple invocation</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L45">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="45"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        result = client.chat.completions.create(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L46">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="46"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            model=<span class="hljs-string">&quot;tiiuae/falcon-180B-chat&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L47">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="47"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            messages=[<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L48">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="48"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->                {<span class="hljs-string">&quot;role&quot;</span>: <span class="hljs-string">&quot;user&quot;</span>, <span class="hljs-string">&quot;content&quot;</span>: <span class="hljs-string">f&quot;Extract book title, description, author names, publisher, and suggest genre from the following text: <span class="hljs-subst">{text}</span> and generate JSON output.&quot;</span>},<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L49">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="49"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            ],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L50">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="50"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L51">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="51"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L52">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="52"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        content = result.choices[<span class="hljs-number">0</span>].message.content<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L53">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="53"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> content<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L54">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="54"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">except</span> Exception <span class="hljs-keyword">as</span> e:<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L55">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="55"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> <span class="hljs-string">f&quot;Error in extracting details: <span class="hljs-subst">{<span class="hljs-built_in">str</span>(e)}</span>&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L56">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="56"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L57">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="57"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Function to handle the combined pipeline</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L58">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="58"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">combined_pipeline</span>(<span class="hljs-params">image</span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L59">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="59"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">try</span>:<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L60">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="60"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-comment"># Run Florence-2 inference on the uploaded image</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L61">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="61"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        extracted_text = run_inference(image)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L62">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="62"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L63">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="63"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-comment"># Extract details from the text using Falcon</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L64">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="64"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        book_details = extract_details(extracted_text)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L65">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="65"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L66">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="66"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> book_details<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L67">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="67"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">except</span> Exception <span class="hljs-keyword">as</span> e:<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L68">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="68"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> <span class="hljs-string">f&quot;Error in combined_pipeline: <span class="hljs-subst">{<span class="hljs-built_in">str</span>(e)}</span>&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L69">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="69"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L70">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="70"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Create the Gradio interface</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L71">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="71"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->interface = gr.Interface(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L72">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="72"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    fn=combined_pipeline,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L73">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="73"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    inputs=gr.Image(<span class="hljs-built_in">type</span>=<span class="hljs-string">&quot;pil&quot;</span>, mirror_webcam=<span class="hljs-literal">False</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L74">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="74"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    outputs=gr.Textbox(),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L75">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="75"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    title=<span class="hljs-string">&quot;Automate Book Inventory using AI&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L76">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="76"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    description=<span class="hljs-string">&quot;Upload an image of a BOOK cover to run Florence-2 inference to extract OCR and understand and prepare structured book information using Falcon.&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L77">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="77"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L78">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="78"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L79">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="79"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Launch the Gradio app</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L80">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="80"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->interface.launch(share=<span class="hljs-literal">True</span>)<!-- HTML_TAG_END --></td>
					</tr></tbody></table></div>
	</div></div></div></div></section></div></main>

	</div>

		<script>
			import("/front/build/kube-0932549/index.js");
			window.moonSha = "kube-0932549/";
		</script>

		<!-- Stripe -->
		<script>
			if (["hf.co", "huggingface.co"].includes(window.location.hostname)) {
				const script = document.createElement("script");
				script.src = "https://js.stripe.com/v3/";
				script.async = true;
				document.head.appendChild(script);
			}
		</script>
	</body>
</html>
