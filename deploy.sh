#!/bin/bash

# RCOA GitHub Pages Deployment Script
# This script helps deploy the RCOA demos to GitHub Pages

set -e

echo "🌾🦀 RCOA GitHub Pages Deployment Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

echo -e "${BLUE}📋 Checking repository status...${NC}"
git status --short

echo ""
echo -e "${YELLOW}🔍 What would you like to do?${NC}"
echo "1. Check deployment status"
echo "2. Trigger manual deployment (workflow_dispatch)"
echo "3. Push to main/master branch (triggers auto-deploy)"
echo "4. View GitHub Pages URL"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "${BLUE}📊 Checking latest workflow runs...${NC}"
        if command -v gh &> /dev/null; then
            gh run list --workflow=deploy-pages.yml --limit 5
        else
            echo "⚠️  GitHub CLI (gh) not installed. Install it to view workflow status."
            echo "Visit: https://cli.github.com/"
        fi
        ;;
    2)
        echo -e "${BLUE}🚀 Triggering manual deployment...${NC}"
        if command -v gh &> /dev/null; then
            gh workflow run deploy-pages.yml
            echo -e "${GREEN}✓ Workflow triggered successfully!${NC}"
            echo "View status with: gh run list --workflow=deploy-pages.yml"
        else
            echo "⚠️  GitHub CLI (gh) not installed. Install it to trigger workflows."
            echo "Visit: https://cli.github.com/"
            echo ""
            echo "Alternative: Trigger manually from GitHub Actions tab:"
            gh_repo=$(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/')
            echo "https://github.com/$gh_repo/actions/workflows/deploy-pages.yml"
        fi
        ;;
    3)
        echo -e "${BLUE}📤 Preparing to push to main branch...${NC}"
        current_branch=$(git branch --show-current)
        echo "Current branch: $current_branch"
        echo ""

        if [[ "$current_branch" != "main" && "$current_branch" != "master" ]]; then
            echo -e "${YELLOW}⚠️  You are not on main/master branch.${NC}"
            read -p "Do you want to switch to main branch? (y/n): " switch_branch
            if [[ "$switch_branch" == "y" ]]; then
                if git show-ref --verify --quiet refs/heads/main; then
                    git checkout main
                elif git show-ref --verify --quiet refs/heads/master; then
                    git checkout master
                else
                    echo "❌ No main or master branch found"
                    exit 1
                fi
            else
                echo "Deployment cancelled."
                exit 0
            fi
        fi

        echo -e "${BLUE}🔄 Pulling latest changes...${NC}"
        git pull origin $(git branch --show-current) || true

        echo -e "${BLUE}📤 Pushing to remote...${NC}"
        git push -u origin $(git branch --show-current)

        echo -e "${GREEN}✓ Pushed successfully! GitHub Pages will deploy automatically.${NC}"
        echo "Monitor deployment: gh run list --workflow=deploy-pages.yml"
        ;;
    4)
        gh_repo=$(git remote get-url origin 2>/dev/null | sed 's/.*github.com[:/]\(.*\)\.git/\1/' || echo "")
        if [[ -n "$gh_repo" ]]; then
            gh_user=$(echo $gh_repo | cut -d'/' -f1)
            gh_repo_name=$(echo $gh_repo | cut -d'/' -f2)
            pages_url="https://${gh_user}.github.io/${gh_repo_name}/"
            echo -e "${GREEN}🌐 GitHub Pages URL:${NC}"
            echo "$pages_url"
            echo ""
            echo "Note: URL will be active after first successful deployment."
        else
            echo "❌ Could not determine repository information"
        fi
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ Done!${NC}"
